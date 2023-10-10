#pragma once

#include <functional>

#include <glm/glm.hpp>

extern void display(std::function<glm::vec4*(glm::ivec2 viewport)> update);
