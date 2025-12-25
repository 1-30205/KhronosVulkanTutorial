#ifndef GLM_API_H
#define GLM_API_H

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

namespace glm
{
    template<typename T, qualifier Q>
    GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookTowardsRH(vec<3, T, Q> const& eye, vec<3, T, Q> const& dir, vec<3, T, Q> const& up)
    {
        vec<3, T, Q> const f(normalize(dir));
        vec<3, T, Q> const s(normalize(cross(f, up)));
        vec<3, T, Q> const u(cross(s, f));

        mat<4, 4, T, Q> Result(1);
        Result[0][0] = s.x;
        Result[1][0] = s.y;
        Result[2][0] = s.z;
        Result[0][1] = u.x;
        Result[1][1] = u.y;
        Result[2][1] = u.z;
        Result[0][2] =-f.x;
        Result[1][2] =-f.y;
        Result[2][2] =-f.z;
        Result[3][0] =-dot(s, eye);
        Result[3][1] =-dot(u, eye);
        Result[3][2] = dot(f, eye);
        return Result;
    }

    template<typename T, qualifier Q>
    GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookTowardsLH(vec<3, T, Q> const& eye, vec<3, T, Q> const& dir, vec<3, T, Q> const& up)
    {
        vec<3, T, Q> const f(normalize(dir));
        vec<3, T, Q> const s(normalize(cross(up, f)));
        vec<3, T, Q> const u(cross(f, s));

        mat<4, 4, T, Q> Result(1);
        Result[0][0] = s.x;
        Result[1][0] = s.y;
        Result[2][0] = s.z;
        Result[0][1] = u.x;
        Result[1][1] = u.y;
        Result[2][1] = u.z;
        Result[0][2] = f.x;
        Result[1][2] = f.y;
        Result[2][2] = f.z;
        Result[3][0] = -dot(s, eye);
        Result[3][1] = -dot(u, eye);
        Result[3][2] = -dot(f, eye);
        return Result;
    }

    template<typename T, qualifier Q>
    GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookTowards(vec<3, T, Q> const& eye, vec<3, T, Q> const& dir, vec<3, T, Q> const& up)
    {
#       if (GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT)
            return lookTowardsLH(eye, dir, up);
#       else
            return lookTowardsRH(eye, dir, up);
#       endif
    }
} // namespace glm

#endif // GLM_API_H