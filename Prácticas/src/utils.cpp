#include "../inc/utils.hpp"

#include <cstddef>        // std::size_t

std::string SplitFilename (const std::string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}
