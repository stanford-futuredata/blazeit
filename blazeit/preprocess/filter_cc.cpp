#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "folly/dynamic.h"
#include "folly/json.h"
#include "clipper.hpp"

class Row {
  const std::string line_;

 public:
  size_t frame_;
  int cc_ind_;
  std::string object_name_;
  float conf_;
  float xmin_, ymin_, xmax_, ymax_;

  Row(const std::string& line) : line_(line) {
    char tmp[1000];
    sscanf(line.c_str(), "%zu%*c%79[^,]%*c%f%*c%f%*c%f%*c%f%*c%f%*c%d",
           &frame_, tmp, &conf_, &xmin_, &ymin_, &xmax_, &ymax_, &cc_ind_);
    object_name_ = tmp;
  }

  bool operator<(const Row& other) {
    if (frame_ != other.frame_) return frame_ < other.frame_;
    if (object_name_ != other.object_name_) return object_name_ < other.object_name_;
    if (cc_ind_ != other.cc_ind_) return cc_ind_ < other.cc_ind_;
    return conf_ < other.conf_;
  }

  float area() const {
    return (xmax_ - xmin_) * (ymax_ - ymin_);
  }

  float intersection(const float oxmin, const float oymin, const float oxmax, const float oymax) {
    if (xmax_ < oxmin || oxmax < xmin_ ||
        ymax_ < oymin || oymax < ymin_)
      return 0.f;
    float xA = std::min(xmax_, oxmax);
    float yA = std::min(ymax_, oymax);
    float xB = std::max(xmin_, oxmin);
    float yB = std::max(ymin_, oymin);
    float interArea = (xA - xB) * (yA - yB);
    float a1 = area();
    return std::max(interArea / a1, 0.f);
  }

  std::pair<float, float> center() const {
    return {(xmax_ + xmin_) / 2, (ymax_ + ymin_) / 2};
  }

  std::string serialize() {
    std::string out(line_);
    while (out.back() != ',')
      out.pop_back();
    out += std::to_string(cc_ind_);
    return out;
  }
};

typedef std::vector<Row *> ConnectedComponent;


// FIXME: name is pretty horrible
namespace cl = ClipperLib;
class ValidBox {
 private:
  bool no_crop_;
  int crop_xmin_, crop_xmax_, crop_ymin_, crop_ymax_;
  std::vector<cl::Path> exclude_regions_;

 public:
  ValidBox(const std::string& json_fname) {
    std::ifstream json_fstream(json_fname);
    const std::string json_txt(
        (std::istreambuf_iterator<char>(json_fstream)),
        std::istreambuf_iterator<char>());
    folly::dynamic parsed = folly::parseJson(json_txt);

    no_crop_ = parsed["crop"].isNull();
    crop_xmin_ = no_crop_ ? 0 : parsed["crop"][0].asInt();
    crop_ymin_ = no_crop_ ? 0 : parsed["crop"][1].asInt();
    crop_xmax_ = no_crop_ ? 0 : parsed["crop"][2].asInt();
    crop_ymax_ = no_crop_ ? 0 : parsed["crop"][3].asInt();

    for (auto poly : parsed["exclude"]) {
      cl::Path p;
      for (auto point : poly) {
        p << cl::IntPoint(point[0].asInt(), point[1].asInt());
      }
      exclude_regions_.push_back(p);
    }
  }

  bool isValid(Row *row, const double min_conf) const {
    if (row->conf_ < min_conf)
      return false;
    cl::Paths box(1);
    box[0] << cl::IntPoint(row->xmin_, row->ymin_) <<
        cl::IntPoint(row->xmin_, row->ymax_) <<
        cl::IntPoint(row->xmax_, row->ymax_) <<
        cl::IntPoint(row->xmax_, row->ymin_);
    const double box_area = row->area();

    cl::Clipper c;
    cl::Paths res;
    c.AddPaths(box, cl::ptSubject, true);
    c.AddPaths(exclude_regions_, cl::ptClip, true);
    c.Execute(cl::ctIntersection, res, cl::pftNonZero, cl::pftNonZero);
    double area = 0.;
    for (auto p: res)
      area += cl::Area(p);
    if (area / box_area > 0.6)
      return false;

    if (no_crop_)
      return true;

    return row->intersection(crop_xmin_, crop_ymin_, crop_xmax_, crop_ymax_) > 0.25;
  }
};

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " MIN_CONF JSON_FNAME IN_FNAME OUT_FNAME" << std::endl;
    return 0;
  }
  const double min_conf = std::stod(argv[1]);
  const std::string json_fname = argv[2];
  const std::string csv_in_fname = argv[3];
  const std::string csv_out_fname = argv[4];
  ValidBox valid_box(json_fname);

  if (csv_in_fname == csv_out_fname) {
    std::cerr << "Input and output fnames cannot be the same." << std::endl;
    return 0;
  }
  std::cout << "Processing: " << csv_in_fname << " " << csv_out_fname << std::endl;

  std::ifstream csv(csv_in_fname);
  std::vector<std::string> lines;

  // Read data
  auto start = std::chrono::high_resolution_clock::now();
  std::copy(std::istream_iterator<std::string>(csv),
            std::istream_iterator<std::string>(),
            std::back_inserter(lines));
  std::vector<Row *> rows(lines.size() - 1);
  #pragma omp parallel for
  for (size_t i = 1; i < lines.size(); i++) {
    rows[i - 1] = new Row(lines[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time to read data, extract to rows: " << elapsed.count() << " s\n";

  size_t nb_inds = 0;
  for (auto row: rows)
    nb_inds = std::max(nb_inds, (size_t) (row->cc_ind_ + 1));

  // Get the ccs
  std::vector<ConnectedComponent> all_ccs(nb_inds);
  for (size_t i = 0; i < rows.size(); i++) {
    all_ccs[rows[i]->cc_ind_].push_back(rows[i]);
  }
  std::vector<Row *> filtered_rows;
  for (size_t i = 0; i < all_ccs.size(); i++) {
    for (auto row: all_ccs[i])
      filtered_rows.push_back(row);
  }
  std::cout << filtered_rows.size() << std::endl;

  // Filter out boxes not in the allowed area
  std::vector<Row *> valid_rows;
  std::vector<bool> is_valid(filtered_rows.size());
  #pragma omp parallel for
  for (size_t i = 0; i < filtered_rows.size(); i++) {
    is_valid[i] = valid_box.isValid(filtered_rows[i], min_conf);
  }
  auto valid_mask = is_valid.begin();
  std::copy_if(filtered_rows.begin(), filtered_rows.end(),
               std::back_inserter(valid_rows),
               [&valid_mask](Row* row) { return *valid_mask++; });
  std::cout << valid_rows.size() << std::endl;

  // Compactify inds
  std::unordered_map<size_t, size_t> ind_map;
  size_t counter = 0;
  for (auto row: valid_rows) {
    if (ind_map.count(row->cc_ind_) > 0)
      continue;
    else
      ind_map[row->cc_ind_] = counter++;
  }
  for (auto row: valid_rows) {
    row->cc_ind_ = ind_map[row->cc_ind_];
  }
  std::sort(valid_rows.begin(), valid_rows.end());

  // Serialize the rows
  std::ofstream out(csv_out_fname);
  out << "frame,object_name,confidence,xmin,ymin,xmax,ymax,ind\n";
  for (auto row: valid_rows) {
    if (row->cc_ind_ < 0)
      continue;
    out << row->serialize() << "\n";
  }
  out.close();
}
