#include "utils.hpp"

#include <string>         // std::string
#include <cstddef>        // std::size_t

void SplitFilename (const std::string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}
