#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <sstream>
#include <unordered_set>

typedef std::vector<std::vector<size_t>> EdgeList;
typedef std::vector<size_t> ConnectedComponent;

// Contains information about row
class Row {
  const std::string line_;

  // FIXME: maybe?
 public:
  size_t frame_;
  size_t graph_ind_;
  int cc_ind_;
  std::string object_name_;
  float conf_;
  float xmin_, ymin_, xmax_, ymax_;

  Row(const std::string& line) : line_(line), cc_ind_(-1) {
    char tmp[1000];
    sscanf(line.c_str(), "%zu%*c%79[^,]%*c%f%*c%f%*c%f%*c%f%*c%f%*c",
           &frame_, tmp, &conf_, &xmin_, &ymin_, &xmax_, &ymax_);
    object_name_ = tmp;
    /*std::istringstream ss(line);
    char c;

    ss >> frame_;
    ss.ignore();
    std::getline(ss, object_name_, ',');
    ss >> conf_ >> c;*/
    /*std::string tmp;
    std::getline(ss, tmp);
    sscanf(tmp.c_str(), "%f%*c%f%*c%f%*c%f", &xmin_, &ymin_, &xmax_, &ymax_);*/
    //ss >> xmin_ >> c >> ymin_ >> c >> xmax_ >> c >> ymax_;

    /*std::cout << frame_ << std::endl;
    std::cout << object_name_ << std::endl;
    std::cout << conf_ << std::endl;
    std::cout << xmin_ << " " <<  ymin_ << " " << xmax_ << " " << ymax_ << std::endl << std::endl;*/
  }

  std::string serialize() {
    return line_ + "," + std::to_string(cc_ind_);
  }

  float area() const {
    return (xmax_ - xmin_) * (ymax_ - ymin_);
  }

  float iou(const Row* other) {
    if (xmax_ < other->xmin_ || other->xmax_ < xmin_ ||
        ymax_ < other->ymin_ || other->ymax_ < ymin_)
      return 0.f;
    float xA = std::min(xmax_, other->xmax_);
    float yA = std::min(ymax_, other->ymax_);
    float xB = std::max(xmin_, other->xmin_);
    float yB = std::max(ymin_, other->ymin_);
    float interArea = (xA - xB) * (yA - yB);
    float a1 = area();
    float a2 = other->area();
    return std::max(interArea / (a1 + a2 - interArea), 0.f);
  }

  bool connected(const Row* const other) {
    if (object_name_ != other->object_name_)
      return false;
    return iou(other) > 0.7;
  }

  bool connected(const Row& other) {
    return connected(&other);
  }
};

std::vector<ConnectedComponent> get_connected_components(EdgeList edges) {
  std::unordered_set<size_t> visited;
  std::vector<ConnectedComponent> ccs;

  auto floodfill = [&edges, &visited](size_t start, ConnectedComponent& cc) {
    std::deque<size_t> to_visit;
    to_visit.push_back(start);
    while (to_visit.size() > 0) {
      size_t cur = to_visit.front();
      to_visit.pop_front();
      if (visited.count(cur) > 0)
        continue;
      visited.insert(cur);
      cc.push_back(cur);
      for (size_t next: edges[cur])
        to_visit.push_back(next);
    }
  };

  for (size_t i = 0; i < edges.size(); i++) {
    if (visited.count(i) > 0)
      continue;
    ConnectedComponent cc;
    floodfill(i, cc);
    ccs.push_back(cc);
  }
  return ccs;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " IN_FNAME OUT_FNAME" << std::endl;
    return 0;
  }
  const std::string csv_in_fname = argv[1];
  const std::string csv_out_fname = argv[2];
  if (csv_in_fname == csv_out_fname) {
    std::cerr << "Input and output fnames cannot be the same." << std::endl;
    return 0;
  }
  std::cerr << "Processing: " << csv_in_fname << " " << csv_out_fname << std::endl;

  std::ifstream csv(csv_in_fname);
  std::vector<std::string> lines;
  auto start = std::chrono::high_resolution_clock::now();
  std::copy(std::istream_iterator<std::string>(csv),
            std::istream_iterator<std::string>(),
            std::back_inserter(lines));
  /*std::copy_n(std::istream_iterator<std::string>(csv),
              10000000,
              std::back_inserter(lines));*/
  std::vector<Row *> rows(lines.size() - 1);
  #pragma omp parallel for
  for (size_t i = 1; i < lines.size(); i++) {
    rows[i - 1] = new Row(lines[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Time to read data, extract to rows: " << elapsed.count() << " s\n";

  // Get frame_to_rows
  size_t nb_frames = 0;
  for (size_t i = 0; i < rows.size(); i++) {
    nb_frames = std::max(nb_frames, rows[i]->frame_ + 1);
    rows[i]->graph_ind_ = i;
  }
  std::vector<std::vector<Row *>> frame_to_rows(nb_frames);
  for (size_t i = 0; i < rows.size(); i++) {
    frame_to_rows[rows[i]->frame_].push_back(rows[i]);
  }

  // get edges
  EdgeList edges(rows.size());
  for (size_t cur_frame = 0; cur_frame < nb_frames; cur_frame++) {
    for (size_t foffset = 1; foffset < 4; foffset++) {
      const size_t next_frame = cur_frame + foffset;
      if (next_frame >= frame_to_rows.size())
        break;
      for (auto cf_row: frame_to_rows[cur_frame]) {
        for (auto nf_row: frame_to_rows[next_frame]) {
          if (cf_row->connected(nf_row)) {
            edges[cf_row->graph_ind_].push_back(nf_row->graph_ind_);
            edges[nf_row->graph_ind_].push_back(cf_row->graph_ind_);
          }
        }
      }
    }
  }

  // Get connected components
  std::vector<ConnectedComponent> all_ccs = get_connected_components(edges);
  std::cout << all_ccs.size() << std::endl;

  auto filter_low_conf = [&rows](ConnectedComponent cc) {
    float max_conf = 0;
    for (auto i: cc)
      max_conf = std::max(max_conf, rows[i]->conf_);
    return max_conf > 0.1;
  };
  std::vector<ConnectedComponent> ccs;
  std::copy_if(all_ccs.begin(), all_ccs.end(), std::back_inserter(ccs), filter_low_conf);

  std::cout << ccs.size() << std::endl;
  // assign cc inds to rows
  for (size_t i = 0; i < ccs.size(); i++) {
    for (auto graph_ind: ccs[i])
      rows[graph_ind]->cc_ind_ = i;
  }

  // Serialize the rows
  std::ofstream out(csv_out_fname);
  out << "frame,object_name,confidence,xmin,ymin,xmax,ymax,ind\n";
  for (auto row: rows) {
    if (row->cc_ind_ < 0)
      continue;
    out << row->serialize() << "\n";
  }
  out.close();
}
