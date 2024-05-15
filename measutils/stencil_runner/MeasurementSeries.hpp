#pragma once
#include <algorithm>
#include <numeric>
#include <vector>

class MeasurementSeries {
public:
  void add(double v) { data.push_back(v); }
  double value() {
    if (data.size() == 0)
      return 0.0;
    if (data.size() == 1)
      return data[0];
    if (data.size() == 2)
      return (data[0] + data[1]) / 2.0;
    std::sort(std::begin(data), std::end(data));
    return std::accumulate(std::begin(data) + 1, std::end(data) - 1, 0.0) /
           (data.size() - 2);
  }
  double median() {
    if (data.size() == 0)
      return 0.0;
    if (data.size() == 1)
      return data[0];
    if (data.size() == 2)
      return (data[0] + data[1]) / 2.0;

    std::sort(std::begin(data), std::end(data));
    if (data.size() % 2 == 0) {
      return (data[data.size() / 2] + data[data.size() / 2 + 1]) / 2;
    }
    return data[data.size() / 2];
  }

  double minValue() {
    std::sort(std::begin(data), std::end(data));
    return *std::begin(data);
  }

  double maxValue() {
    std::sort(std::begin(data), std::end(data));
    return data.back();
  }
  double spread() {
    if (data.size() <= 1)
      return 0.0;
    if (data.size() == 2)
      return abs(data[0] - data[1]) / value();
    std::sort(std::begin(data), std::end(data));
    return abs(*(std::begin(data)) - *(std::end(data) - 1)) / value();
  }
  int count() { return data.size(); }

  auto begin() { return data.begin(); }
  auto end() { return data.end(); }

private:
  std::vector<double> data;
};
