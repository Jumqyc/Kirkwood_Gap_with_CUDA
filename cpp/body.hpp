#pragma once

#include "vec3.hpp"

// A massive body, in the heliocentric frame with the Sun pinned at the origin.
//
// Note: v and a are per *step*, not per day -- dt is folded into G, so a
// velocity written here means AU per step and changes meaning if dt changes.
struct Planet
{
    double mass;
    Vec3 r, v, a;
};
