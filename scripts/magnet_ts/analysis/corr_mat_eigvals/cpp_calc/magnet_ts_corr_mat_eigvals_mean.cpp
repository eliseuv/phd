#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

  // Get filename from arguments
  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i++) {
    filenames.push_back(argv[i]);
  }

  // Loop on datafiles
  for (auto &filename : filenames) {

    // Open datafile
    std::cout << "Reading: " << filename << std::endl;
    std::ifstream datafile(filename);

    // Vector of eigenvalues
    std::vector<double> eigvals;

    // We want to read all lines of the file
    std::string line{};
    while (datafile && getline(datafile, line)) {
      eigvals.push_back(std::stod(line));
    }
    std::cout << "Eigenvalues count: " << eigvals.size() << std::endl;

    // Calculate average value
    double sum = 0.0;
    for (auto &eigval : eigvals) {
      // std::cout << eigval << std::endl;
      sum += eigval;
      std::cout << sum << std::endl;
    }
    double avg = sum / eigvals.size();
    std::cout << "Average eigenvalue: " << avg << std::endl;
  }

  return 0;
}
