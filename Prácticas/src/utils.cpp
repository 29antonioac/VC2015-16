#include "../inc/utils.hpp"

#include <cstddef>        // std::size_t
#include <cmath>
#include <iostream>

using namespace std;

string SplitFilename (const string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}
