// #ifndef LINMATH_H
// #define LINMATH_H

// #include <math.h>

// #define LINMATH_H_DEFINE_VEC(n)                                                \
//     typedef float vec##n[n];                                                   \
//     static inline void vec##n##_add(vec##n r, vec##n const a, vec##n const b)  \
//     {                                                                          \
//         int i;                                                                 \
//         for (i = 0; i < n; ++i)                                                \
//             r[i] = a[i] + b[i];                                                \
//     }                                                                          \
//     static inline void vec##n##_sub(vec##n r, vec##n const a, vec##n const b)  \
//     {                                                                          \
//         int i;                                                                 \
//         for (i = 0; i < n; ++i)                                                \
//             r[i] = a[i] - b[i];                                                \
//     }                                                                          \
//     static inline void vec##n##_scale(vec##n r, vec##n const v, float const s) \
//     {                                                                          \
//         int i;                                                                 \
//         for (i = 0; i < n; ++i)                                                \
//             r[i] = v[i] * s;                                                   \
//     }                                                                          \
//     static inline float vec##n##_mul_inner(vec##n const a, vec##n const b)     \
//     {                                                                          \
//         float p = 0.;                                                          \
//         int i;                                                                 \
//         for (i = 0; i < n; ++i)                                                \
//             p += b[i] * a[i];                                                  \
//         return p;                                                              \
//     }                                                                          \
//     static inline float vec##n##_len(vec##n const v)                           \
//     {                                                                          \
//         return (float)sqrt(vec##n##_mul_inner(v, v));                          \
//     }                                                                          \
//     static inline void vec##n##_norm(vec##n r, vec##n const v)                 \
//     {                                                                          \
//         float k = 1.f / vec##n##_len(v);                                       \
//         vec##n##_scale(r, v, k);                                               \
//     }

// LINMATH_H_DEFINE_VEC(2)
// LINMATH_H_DEFINE_VEC(3)
// LINMATH_H_DEFINE_VEC(4)

// static inline void vec3_mul_cross(vec3 r, vec3 const a, vec3 const b)
// {
//     r[0] = a[1] * b[2] - a[2] * b[1];
//     r[1] = a[2] * b[0] - a[0] * b[2];
//     r[2] = a[0] * b[1] - a[1] * b[0];
// }

// static inline void vec3_reflect(vec3 r, vec3 const v, vec3 const n)
// {
//     float p = 2.f * vec3_mul_inner(v, n);
//     int i;
//     for (i = 0; i < 3; ++i)
//         r[i] = v[i] - p * n[i];
// }

// static inline void vec4_mul_cross(vec4 r, vec4 a, vec4 b)
// {
//     r[0] = a[1] * b[2] - a[2] * b[1];
//     r[1] = a[2] * b[0] - a[0] * b[2];
//     r[2] = a[0] * b[1] - a[1] * b[0];
//     r[3] = 1.f;
// }

// static inline void vec4_reflect(vec4 r, vec4 v, vec4 n)
// {
//     float p = 2.f * vec4_mul_inner(v, n);
//     int i;
//     for (i = 0; i < 4; ++i)
//         r[i] = v[i] - p * n[i];
// }

// typedef vec4 mat4x4[4];
// static inline void mat4x4_identity(mat4x4 M)
// {
//     int i, j;
//     for (i = 0; i < 4; ++i)
//         for (j = 0; j < 4; ++j)
//             M[i][j] = i == j ? 1.f : 0.f;
// }
// static inline void mat4x4_dup(mat4x4 M, mat4x4 N)
// {
//     int i, j;
//     for (i = 0; i < 4; ++i)
//         for (j = 0; j < 4; ++j)
//             M[i][j] = N[i][j];
// }
// static inline void mat4x4_row(vec4 r, mat4x4 M, int i)
// {
//     int k;
//     for (k = 0; k < 4; ++k)
//         r[k] = M[k][i];
// }
// static inline void mat4x4_col(vec4 r, mat4x4 M, int i)
// {
//     int k;
//     for (k = 0; k < 4; ++k)
//         r[k] = M[i][k];
// }
// static inline void mat4x4_transpose(mat4x4 M, mat4x4 N)
// {
//     int i, j;
//     for (j = 0; j < 4; ++j)
//         for (i = 0; i < 4; ++i)
//             M[i][j] = N[j][i];
// }
// static inline void mat4x4_add(mat4x4 M, mat4x4 a, mat4x4 b)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         vec4_add(M[i], a[i], b[i]);
// }
// static inline void mat4x4_sub(mat4x4 M, mat4x4 a, mat4x4 b)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         vec4_sub(M[i], a[i], b[i]);
// }
// static inline void mat4x4_scale(mat4x4 M, mat4x4 a, float k)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         vec4_scale(M[i], a[i], k);
// }
// static inline void mat4x4_scale_aniso(mat4x4 M, mat4x4 a, float x, float y, float z)
// {
//     int i;
//     vec4_scale(M[0], a[0], x);
//     vec4_scale(M[1], a[1], y);
//     vec4_scale(M[2], a[2], z);
//     for (i = 0; i < 4; ++i)
//     {
//         M[3][i] = a[3][i];
//     }
// }
// static inline void mat4x4_mul(mat4x4 M, mat4x4 a, mat4x4 b)
// {
//     mat4x4 temp;
//     int k, r, c;
//     for (c = 0; c < 4; ++c)
//         for (r = 0; r < 4; ++r)
//         {
//             temp[c][r] = 0.f;
//             for (k = 0; k < 4; ++k)
//                 temp[c][r] += a[k][r] * b[c][k];
//         }
//     mat4x4_dup(M, temp);
// }
// static inline void mat4x4_mul_vec4(vec4 r, mat4x4 M, vec4 v)
// {
//     int i, j;
//     for (j = 0; j < 4; ++j)
//     {
//         r[j] = 0.f;
//         for (i = 0; i < 4; ++i)
//             r[j] += M[i][j] * v[i];
//     }
// }
// static inline void mat4x4_translate(mat4x4 T, float x, float y, float z)
// {
//     mat4x4_identity(T);
//     T[3][0] = x;
//     T[3][1] = y;
//     T[3][2] = z;
// }
// static inline void mat4x4_translate_in_place(mat4x4 M, float x, float y, float z)
// {
//     vec4 t = {x, y, z, 0};
//     vec4 r;
//     int i;
//     for (i = 0; i < 4; ++i)
//     {
//         mat4x4_row(r, M, i);
//         M[3][i] += vec4_mul_inner(r, t);
//     }
// }
// static inline void mat4x4_from_vec3_mul_outer(mat4x4 M, vec3 a, vec3 b)
// {
//     int i, j;
//     for (i = 0; i < 4; ++i)
//         for (j = 0; j < 4; ++j)
//             M[i][j] = i < 3 && j < 3 ? a[i] * b[j] : 0.f;
// }
// static inline void mat4x4_rotate(mat4x4 R, mat4x4 M, float x, float y, float z, float angle)
// {
//     float s = sinf(angle);
//     float c = cosf(angle);
//     vec3 u = {x, y, z};

//     if (vec3_len(u) > 1e-4)
//     {
//         mat4x4 T, C, S = {{0}};

//         vec3_norm(u, u);
//         mat4x4_from_vec3_mul_outer(T, u, u);

//         S[1][2] = u[0];
//         S[2][1] = -u[0];
//         S[2][0] = u[1];
//         S[0][2] = -u[1];
//         S[0][1] = u[2];
//         S[1][0] = -u[2];

//         mat4x4_scale(S, S, s);

//         mat4x4_identity(C);
//         mat4x4_sub(C, C, T);

//         mat4x4_scale(C, C, c);

//         mat4x4_add(T, T, C);
//         mat4x4_add(T, T, S);

//         T[3][3] = 1.;
//         mat4x4_mul(R, M, T);
//     }
//     else
//     {
//         mat4x4_dup(R, M);
//     }
// }
// static inline void mat4x4_rotate_X(mat4x4 Q, mat4x4 M, float angle)
// {
//     float s = sinf(angle);
//     float c = cosf(angle);
//     mat4x4 R = {
//         {1.f, 0.f, 0.f, 0.f},
//         {0.f, c, s, 0.f},
//         {0.f, -s, c, 0.f},
//         {0.f, 0.f, 0.f, 1.f}};
//     mat4x4_mul(Q, M, R);
// }
// static inline void mat4x4_rotate_Y(mat4x4 Q, mat4x4 M, float angle)
// {
//     float s = sinf(angle);
//     float c = cosf(angle);
//     mat4x4 R = {
//         {c, 0.f, -s, 0.f},
//         {0.f, 1.f, 0.f, 0.f},
//         {s, 0.f, c, 0.f},
//         {0.f, 0.f, 0.f, 1.f}};
//     mat4x4_mul(Q, M, R);
// }
// static inline void mat4x4_rotate_Z(mat4x4 Q, mat4x4 M, float angle)
// {
//     float s = sinf(angle);
//     float c = cosf(angle);
//     mat4x4 R = {
//         {c, s, 0.f, 0.f},
//         {-s, c, 0.f, 0.f},
//         {0.f, 0.f, 1.f, 0.f},
//         {0.f, 0.f, 0.f, 1.f}};
//     mat4x4_mul(Q, M, R);
// }
// static inline void mat4x4_invert(mat4x4 T, mat4x4 M)
// {
//     float idet;
//     float s[6];
//     float c[6];
//     s[0] = M[0][0] * M[1][1] - M[1][0] * M[0][1];
//     s[1] = M[0][0] * M[1][2] - M[1][0] * M[0][2];
//     s[2] = M[0][0] * M[1][3] - M[1][0] * M[0][3];
//     s[3] = M[0][1] * M[1][2] - M[1][1] * M[0][2];
//     s[4] = M[0][1] * M[1][3] - M[1][1] * M[0][3];
//     s[5] = M[0][2] * M[1][3] - M[1][2] * M[0][3];

//     c[0] = M[2][0] * M[3][1] - M[3][0] * M[2][1];
//     c[1] = M[2][0] * M[3][2] - M[3][0] * M[2][2];
//     c[2] = M[2][0] * M[3][3] - M[3][0] * M[2][3];
//     c[3] = M[2][1] * M[3][2] - M[3][1] * M[2][2];
//     c[4] = M[2][1] * M[3][3] - M[3][1] * M[2][3];
//     c[5] = M[2][2] * M[3][3] - M[3][2] * M[2][3];

//     /* Assumes it is invertible */
//     idet = 1.0f / (s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]);

//     T[0][0] = (M[1][1] * c[5] - M[1][2] * c[4] + M[1][3] * c[3]) * idet;
//     T[0][1] = (-M[0][1] * c[5] + M[0][2] * c[4] - M[0][3] * c[3]) * idet;
//     T[0][2] = (M[3][1] * s[5] - M[3][2] * s[4] + M[3][3] * s[3]) * idet;
//     T[0][3] = (-M[2][1] * s[5] + M[2][2] * s[4] - M[2][3] * s[3]) * idet;

//     T[1][0] = (-M[1][0] * c[5] + M[1][2] * c[2] - M[1][3] * c[1]) * idet;
//     T[1][1] = (M[0][0] * c[5] - M[0][2] * c[2] + M[0][3] * c[1]) * idet;
//     T[1][2] = (-M[3][0] * s[5] + M[3][2] * s[2] - M[3][3] * s[1]) * idet;
//     T[1][3] = (M[2][0] * s[5] - M[2][2] * s[2] + M[2][3] * s[1]) * idet;

//     T[2][0] = (M[1][0] * c[4] - M[1][1] * c[2] + M[1][3] * c[0]) * idet;
//     T[2][1] = (-M[0][0] * c[4] + M[0][1] * c[2] - M[0][3] * c[0]) * idet;
//     T[2][2] = (M[3][0] * s[4] - M[3][1] * s[2] + M[3][3] * s[0]) * idet;
//     T[2][3] = (-M[2][0] * s[4] + M[2][1] * s[2] - M[2][3] * s[0]) * idet;

//     T[3][0] = (-M[1][0] * c[3] + M[1][1] * c[1] - M[1][2] * c[0]) * idet;
//     T[3][1] = (M[0][0] * c[3] - M[0][1] * c[1] + M[0][2] * c[0]) * idet;
//     T[3][2] = (-M[3][0] * s[3] + M[3][1] * s[1] - M[3][2] * s[0]) * idet;
//     T[3][3] = (M[2][0] * s[3] - M[2][1] * s[1] + M[2][2] * s[0]) * idet;
// }
// static inline void mat4x4_orthonormalize(mat4x4 R, mat4x4 M)
// {
//     float s = 1.;
//     vec3 h;

//     mat4x4_dup(R, M);
//     vec3_norm(R[2], R[2]);

//     s = vec3_mul_inner(R[1], R[2]);
//     vec3_scale(h, R[2], s);
//     vec3_sub(R[1], R[1], h);
//     vec3_norm(R[2], R[2]);

//     s = vec3_mul_inner(R[1], R[2]);
//     vec3_scale(h, R[2], s);
//     vec3_sub(R[1], R[1], h);
//     vec3_norm(R[1], R[1]);

//     s = vec3_mul_inner(R[0], R[1]);
//     vec3_scale(h, R[1], s);
//     vec3_sub(R[0], R[0], h);
//     vec3_norm(R[0], R[0]);
// }

// static inline void mat4x4_frustum(mat4x4 M, float l, float r, float b, float t, float n, float f)
// {
//     M[0][0] = 2.f * n / (r - l);
//     M[0][1] = M[0][2] = M[0][3] = 0.f;

//     M[1][1] = 2.f * n / (t - b);
//     M[1][0] = M[1][2] = M[1][3] = 0.f;

//     M[2][0] = (r + l) / (r - l);
//     M[2][1] = (t + b) / (t - b);
//     M[2][2] = -(f + n) / (f - n);
//     M[2][3] = -1.f;

//     M[3][2] = -2.f * (f * n) / (f - n);
//     M[3][0] = M[3][1] = M[3][3] = 0.f;
// }
// static inline void mat4x4_ortho(mat4x4 M, float l, float r, float b, float t, float n, float f)
// {
//     M[0][0] = 2.f / (r - l);
//     M[0][1] = M[0][2] = M[0][3] = 0.f;

//     M[1][1] = 2.f / (t - b);
//     M[1][0] = M[1][2] = M[1][3] = 0.f;

//     M[2][2] = -2.f / (f - n);
//     M[2][0] = M[2][1] = M[2][3] = 0.f;

//     M[3][0] = -(r + l) / (r - l);
//     M[3][1] = -(t + b) / (t - b);
//     M[3][2] = -(f + n) / (f - n);
//     M[3][3] = 1.f;
// }
// static inline void mat4x4_perspective(mat4x4 m, float y_fov, float aspect, float n, float f)
// {
//     /* NOTE: Degrees are an unhandy unit to work with.
//      * linmath.h uses radians for everything! */
//     float const a = 1.f / (float)tan(y_fov / 2.f);

//     m[0][0] = a / aspect;
//     m[0][1] = 0.f;
//     m[0][2] = 0.f;
//     m[0][3] = 0.f;

//     m[1][0] = 0.f;
//     m[1][1] = a;
//     m[1][2] = 0.f;
//     m[1][3] = 0.f;

//     m[2][0] = 0.f;
//     m[2][1] = 0.f;
//     m[2][2] = -((f + n) / (f - n));
//     m[2][3] = -1.f;

//     m[3][0] = 0.f;
//     m[3][1] = 0.f;
//     m[3][2] = -((2.f * f * n) / (f - n));
//     m[3][3] = 0.f;
// }
// static inline void mat4x4_look_at(mat4x4 m, vec3 eye, vec3 center, vec3 up)
// {
//     /* Adapted from Android's OpenGL Matrix.java.                        */
//     /* See the OpenGL GLUT documentation for gluLookAt for a description */
//     /* of the algorithm. We implement it in a straightforward way:       */

//     /* TODO: The negation of of can be spared by swapping the order of
//      *       operands in the following cross products in the right way. */
//     vec3 f;
//     vec3 s;
//     vec3 t;

//     vec3_sub(f, center, eye);
//     vec3_norm(f, f);

//     vec3_mul_cross(s, f, up);
//     vec3_norm(s, s);

//     vec3_mul_cross(t, s, f);

//     m[0][0] = s[0];
//     m[0][1] = t[0];
//     m[0][2] = -f[0];
//     m[0][3] = 0.f;

//     m[1][0] = s[1];
//     m[1][1] = t[1];
//     m[1][2] = -f[1];
//     m[1][3] = 0.f;

//     m[2][0] = s[2];
//     m[2][1] = t[2];
//     m[2][2] = -f[2];
//     m[2][3] = 0.f;

//     m[3][0] = 0.f;
//     m[3][1] = 0.f;
//     m[3][2] = 0.f;
//     m[3][3] = 1.f;

//     mat4x4_translate_in_place(m, -eye[0], -eye[1], -eye[2]);
// }

// typedef float quat[4];
// static inline void quat_identity(quat q)
// {
//     q[0] = q[1] = q[2] = 0.f;
//     q[3] = 1.f;
// }
// static inline void quat_add(quat r, quat a, quat b)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         r[i] = a[i] + b[i];
// }
// static inline void quat_sub(quat r, quat a, quat b)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         r[i] = a[i] - b[i];
// }
// static inline void quat_mul(quat r, quat p, quat q)
// {
//     vec3 w;
//     vec3_mul_cross(r, p, q);
//     vec3_scale(w, p, q[3]);
//     vec3_add(r, r, w);
//     vec3_scale(w, q, p[3]);
//     vec3_add(r, r, w);
//     r[3] = p[3] * q[3] - vec3_mul_inner(p, q);
// }
// static inline void quat_scale(quat r, quat v, float s)
// {
//     int i;
//     for (i = 0; i < 4; ++i)
//         r[i] = v[i] * s;
// }
// static inline float quat_inner_product(quat a, quat b)
// {
//     float p = 0.f;
//     int i;
//     for (i = 0; i < 4; ++i)
//         p += b[i] * a[i];
//     return p;
// }
// static inline void quat_conj(quat r, quat q)
// {
//     int i;
//     for (i = 0; i < 3; ++i)
//         r[i] = -q[i];
//     r[3] = q[3];
// }
// static inline void quat_rotate(quat r, float angle, vec3 axis)
// {
//     int i;
//     vec3 v;
//     vec3_scale(v, axis, sinf(angle / 2));
//     for (i = 0; i < 3; ++i)
//         r[i] = v[i];
//     r[3] = cosf(angle / 2);
// }
// #define quat_norm vec4_norm
// static inline void quat_mul_vec3(vec3 r, quat q, vec3 v)
// {
//     /*
//      * Method by Fabian 'ryg' Giessen (of Farbrausch)
//     t = 2 * cross(q.xyz, v)
//     v' = v + q.w * t + cross(q.xyz, t)
//      */
//     vec3 t = {q[0], q[1], q[2]};
//     vec3 u = {q[0], q[1], q[2]};

//     vec3_mul_cross(t, t, v);
//     vec3_scale(t, t, 2);

//     vec3_mul_cross(u, u, t);
//     vec3_scale(t, t, q[3]);

//     vec3_add(r, v, t);
//     vec3_add(r, r, u);
// }
// static inline void mat4x4_from_quat(mat4x4 M, quat q)
// {
//     float a = q[3];
//     float b = q[0];
//     float c = q[1];
//     float d = q[2];
//     float a2 = a * a;
//     float b2 = b * b;
//     float c2 = c * c;
//     float d2 = d * d;

//     M[0][0] = a2 + b2 - c2 - d2;
//     M[0][1] = 2.f * (b * c + a * d);
//     M[0][2] = 2.f * (b * d - a * c);
//     M[0][3] = 0.f;

//     M[1][0] = 2 * (b * c - a * d);
//     M[1][1] = a2 - b2 + c2 - d2;
//     M[1][2] = 2.f * (c * d + a * b);
//     M[1][3] = 0.f;

//     M[2][0] = 2.f * (b * d + a * c);
//     M[2][1] = 2.f * (c * d - a * b);
//     M[2][2] = a2 - b2 - c2 + d2;
//     M[2][3] = 0.f;

//     M[3][0] = M[3][1] = M[3][2] = 0.f;
//     M[3][3] = 1.f;
// }

// static inline void mat4x4o_mul_quat(mat4x4 R, mat4x4 M, quat q)
// {
//     /*  XXX: The way this is written only works for othogonal matrices. */
//     /* TODO: Take care of non-orthogonal case. */
//     quat_mul_vec3(R[0], q, M[0]);
//     quat_mul_vec3(R[1], q, M[1]);
//     quat_mul_vec3(R[2], q, M[2]);

//     R[3][0] = R[3][1] = R[3][2] = 0.f;
//     R[3][3] = 1.f;
// }
// static inline void quat_from_mat4x4(quat q, mat4x4 M)
// {
//     float r = 0.f;
//     int i;

//     int perm[] = {0, 1, 2, 0, 1};
//     int *p = perm;

//     for (i = 0; i < 3; i++)
//     {
//         float m = M[i][i];
//         if (m < r)
//             continue;
//         m = r;
//         p = &perm[i];
//     }

//     r = (float)sqrt(1.f + M[p[0]][p[0]] - M[p[1]][p[1]] - M[p[2]][p[2]]);

//     if (r < 1e-6)
//     {
//         q[0] = 1.f;
//         q[1] = q[2] = q[3] = 0.f;
//         return;
//     }

//     q[0] = r / 2.f;
//     q[1] = (M[p[0]][p[1]] - M[p[1]][p[0]]) / (2.f * r);
//     q[2] = (M[p[2]][p[0]] - M[p[0]][p[2]]) / (2.f * r);
//     q[3] = (M[p[2]][p[1]] - M[p[1]][p[2]]) / (2.f * r);
// }

// #endif

// /*****************************************************************************
//  * Wave Simulation in OpenGL
//  * (C) 2002 Jakob Thomsen
//  * http://home.in.tum.de/~thomsen
//  * Modified for GLFW by Sylvain Hellegouarch - sh@programmationworld.com
//  * Modified for variable frame rate by Marcus Geelnard
//  * 2003-Jan-31: Minor cleanups and speedups / MG
//  * 2010-10-24: Formatting and cleanup - Camilla Löwy
//  *****************************************************************************/

// #if defined(_MSC_VER)
// // Make MS math.h define M_PI
// #define _USE_MATH_DEFINES
// #endif

// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>

// #include <glad/glad.h>
// #define GLFW_INCLUDE_NONE
// #include <GLFW/glfw3.h>

// // Maximum delta T to allow for differential calculations
// #define MAX_DELTA_T 0.01

// // Animation speed (10.0 looks good)
// #define ANIMATION_SPEED 10.0

// GLfloat alpha = 210.f, beta = -70.f;
// GLfloat zoom = 2.f;

// double cursorX;
// double cursorY;

// struct Vertex
// {
//     GLfloat x, y, z;
//     GLfloat r, g, b;
// };

// #define GRIDW 50
// #define GRIDH 50
// #define VERTEXNUM (GRIDW * GRIDH)

// #define QUADW (GRIDW - 1)
// #define QUADH (GRIDH - 1)
// #define QUADNUM (QUADW * QUADH)

// GLuint quad[4 * QUADNUM];
// struct Vertex vertex[VERTEXNUM];

// /* The grid will look like this:
//  *
//  *      3   4   5
//  *      *---*---*
//  *      |   |   |
//  *      | 0 | 1 |
//  *      |   |   |
//  *      *---*---*
//  *      0   1   2
//  */

// //========================================================================
// // Initialize grid geometry
// //========================================================================

// void init_vertices(void)
// {
//     int x, y, p;

//     // Place the vertices in a grid
//     for (y = 0; y < GRIDH; y++)
//     {
//         for (x = 0; x < GRIDW; x++)
//         {
//             p = y * GRIDW + x;

//             vertex[p].x = (GLfloat)(x - GRIDW / 2) / (GLfloat)(GRIDW / 2);
//             vertex[p].y = (GLfloat)(y - GRIDH / 2) / (GLfloat)(GRIDH / 2);
//             vertex[p].z = 0;

//             if ((x % 4 < 2) ^ (y % 4 < 2))
//                 vertex[p].r = 0.0;
//             else
//                 vertex[p].r = 1.0;

//             vertex[p].g = (GLfloat)y / (GLfloat)GRIDH;
//             vertex[p].b = 1.f - ((GLfloat)x / (GLfloat)GRIDW + (GLfloat)y / (GLfloat)GRIDH) / 2.f;
//         }
//     }

//     for (y = 0; y < QUADH; y++)
//     {
//         for (x = 0; x < QUADW; x++)
//         {
//             p = 4 * (y * QUADW + x);

//             quad[p + 0] = y * GRIDW + x;           // Some point
//             quad[p + 1] = y * GRIDW + x + 1;       // Neighbor at the right side
//             quad[p + 2] = (y + 1) * GRIDW + x + 1; // Upper right neighbor
//             quad[p + 3] = (y + 1) * GRIDW + x;     // Upper neighbor
//         }
//     }
// }

// double dt;
// double p[GRIDW][GRIDH];
// double vx[GRIDW][GRIDH], vy[GRIDW][GRIDH];
// double ax[GRIDW][GRIDH], ay[GRIDW][GRIDH];

// //========================================================================
// // Initialize grid
// //========================================================================

// void init_grid(void)
// {
//     int x, y;
//     double dx, dy, d;

//     for (y = 0; y < GRIDH; y++)
//     {
//         for (x = 0; x < GRIDW; x++)
//         {
//             dx = (double)(x - GRIDW / 2);
//             dy = (double)(y - GRIDH / 2);
//             d = sqrt(dx * dx + dy * dy);
//             if (d < 0.1 * (double)(GRIDW / 2))
//             {
//                 d = d * 10.0;
//                 p[x][y] = -cos(d * (M_PI / (double)(GRIDW * 4))) * 100.0;
//             }
//             else
//                 p[x][y] = 0.0;

//             vx[x][y] = 0.0;
//             vy[x][y] = 0.0;
//         }
//     }
// }

// //========================================================================
// // Draw scene
// //========================================================================

// void draw_scene(GLFWwindow *window)
// {
//     // Clear the color and depth buffers
//     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//     // We don't want to modify the projection matrix
//     glMatrixMode(GL_MODELVIEW);
//     glLoadIdentity();

//     // Move back
//     glTranslatef(0.0, 0.0, -zoom);
//     // Rotate the view
//     glRotatef(beta, 1.0, 0.0, 0.0);
//     glRotatef(alpha, 0.0, 0.0, 1.0);

//     glDrawElements(GL_QUADS, 4 * QUADNUM, GL_UNSIGNED_INT, quad);

//     glfwSwapBuffers(window);
// }

// //========================================================================
// // Initialize Miscellaneous OpenGL state
// //========================================================================

// void init_opengl(void)
// {
//     // Use Gouraud (smooth) shading
//     glShadeModel(GL_SMOOTH);

//     // Switch on the z-buffer
//     glEnable(GL_DEPTH_TEST);

//     glEnableClientState(GL_VERTEX_ARRAY);
//     glEnableClientState(GL_COLOR_ARRAY);
//     glVertexPointer(3, GL_FLOAT, sizeof(struct Vertex), vertex);
//     glColorPointer(3, GL_FLOAT, sizeof(struct Vertex), &vertex[0].r); // Pointer to the first color

//     glPointSize(2.0);

//     // Background color is black
//     glClearColor(0, 0, 0, 0);
// }

// //========================================================================
// // Modify the height of each vertex according to the pressure
// //========================================================================

// void adjust_grid(void)
// {
//     int pos;
//     int x, y;

//     for (y = 0; y < GRIDH; y++)
//     {
//         for (x = 0; x < GRIDW; x++)
//         {
//             pos = y * GRIDW + x;
//             vertex[pos].z = (float)(p[x][y] * (1.0 / 50.0));
//         }
//     }
// }

// //========================================================================
// // Calculate wave propagation
// //========================================================================

// void calc_grid(void)
// {
//     int x, y, x2, y2;
//     double time_step = dt * ANIMATION_SPEED;

//     // Compute accelerations
//     for (x = 0; x < GRIDW; x++)
//     {
//         x2 = (x + 1) % GRIDW;
//         for (y = 0; y < GRIDH; y++)
//             ax[x][y] = p[x][y] - p[x2][y];
//     }

//     for (y = 0; y < GRIDH; y++)
//     {
//         y2 = (y + 1) % GRIDH;
//         for (x = 0; x < GRIDW; x++)
//             ay[x][y] = p[x][y] - p[x][y2];
//     }

//     // Compute speeds
//     for (x = 0; x < GRIDW; x++)
//     {
//         for (y = 0; y < GRIDH; y++)
//         {
//             vx[x][y] = vx[x][y] + ax[x][y] * time_step;
//             vy[x][y] = vy[x][y] + ay[x][y] * time_step;
//         }
//     }

//     // Compute pressure
//     for (x = 1; x < GRIDW; x++)
//     {
//         x2 = x - 1;
//         for (y = 1; y < GRIDH; y++)
//         {
//             y2 = y - 1;
//             p[x][y] = p[x][y] + (vx[x2][y] - vx[x][y] + vy[x][y2] - vy[x][y]) * time_step;
//         }
//     }
// }

// //========================================================================
// // Print errors
// //========================================================================

// static void error_callback(int error, const char *description)
// {
//     fprintf(stderr, "Error: %s\n", description);
// }

// //========================================================================
// // Handle key strokes
// //========================================================================

// void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
// {
//     if (action != GLFW_PRESS)
//         return;

//     switch (key)
//     {
//     case GLFW_KEY_ESCAPE:
//         glfwSetWindowShouldClose(window, GLFW_TRUE);
//         break;
//     case GLFW_KEY_SPACE:
//         init_grid();
//         break;
//     case GLFW_KEY_LEFT:
//         alpha += 5;
//         break;
//     case GLFW_KEY_RIGHT:
//         alpha -= 5;
//         break;
//     case GLFW_KEY_UP:
//         beta -= 5;
//         break;
//     case GLFW_KEY_DOWN:
//         beta += 5;
//         break;
//     case GLFW_KEY_PAGE_UP:
//         zoom -= 0.25f;
//         if (zoom < 0.f)
//             zoom = 0.f;
//         break;
//     case GLFW_KEY_PAGE_DOWN:
//         zoom += 0.25f;
//         break;
//     default:
//         break;
//     }
// }

// //========================================================================
// // Callback function for mouse button events
// //========================================================================

// void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
// {
//     if (button != GLFW_MOUSE_BUTTON_LEFT)
//         return;

//     if (action == GLFW_PRESS)
//     {
//         glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
//         glfwGetCursorPos(window, &cursorX, &cursorY);
//     }
//     else
//         glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
// }

// //========================================================================
// // Callback function for cursor motion events
// //========================================================================

// void cursor_position_callback(GLFWwindow *window, double x, double y)
// {
//     if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED)
//     {
//         alpha += (GLfloat)(x - cursorX) / 10.f;
//         beta += (GLfloat)(y - cursorY) / 10.f;

//         cursorX = x;
//         cursorY = y;
//     }
// }

// //========================================================================
// // Callback function for scroll events
// //========================================================================

// void scroll_callback(GLFWwindow *window, double x, double y)
// {
//     zoom += (float)y / 4.f;
//     if (zoom < 0)
//         zoom = 0;
// }

// //========================================================================
// // Callback function for framebuffer resize events
// //========================================================================

// void framebuffer_size_callback(GLFWwindow *window, int width, int height)
// {
//     float ratio = 1.f;
//     mat4x4 projection;

//     if (height > 0)
//         ratio = (float)width / (float)height;

//     // Setup viewport
//     glViewport(0, 0, width, height);

//     // Change to the projection matrix and set our viewing volume
//     glMatrixMode(GL_PROJECTION);
//     mat4x4_perspective(projection,
//                        60.f * (float)M_PI / 180.f,
//                        ratio,
//                        1.f, 1024.f);
//     glLoadMatrixf((const GLfloat *)projection);
// }

// //========================================================================
// // main
// //========================================================================

// int main(int argc, char *argv[])
// {
//     GLFWwindow *window;
//     double t, dt_total, t_old;
//     int width, height;

//     glfwSetErrorCallback(error_callback);

//     if (!glfwInit())
//         exit(EXIT_FAILURE);

//     window = glfwCreateWindow(640, 480, "Wave Simulation", NULL, NULL);
//     if (!window)
//     {
//         glfwTerminate();
//         exit(EXIT_FAILURE);
//     }

//     glfwSetKeyCallback(window, key_callback);
//     glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
//     glfwSetMouseButtonCallback(window, mouse_button_callback);
//     glfwSetCursorPosCallback(window, cursor_position_callback);
//     glfwSetScrollCallback(window, scroll_callback);

//     glfwMakeContextCurrent(window);
//     gladLoadGL();
//     glfwSwapInterval(1);

//     glfwGetFramebufferSize(window, &width, &height);
//     framebuffer_size_callback(window, width, height);

//     // Initialize OpenGL
//     init_opengl();

//     // Initialize simulation
//     init_vertices();
//     init_grid();
//     adjust_grid();

//     // Initialize timer
//     t_old = glfwGetTime() - 0.01;

//     while (!glfwWindowShouldClose(window))
//     {
//         t = glfwGetTime();
//         dt_total = t - t_old;
//         t_old = t;

//         // Safety - iterate if dt_total is too large
//         while (dt_total > 0.f)
//         {
//             // Select iteration time step
//             dt = dt_total > MAX_DELTA_T ? MAX_DELTA_T : dt_total;
//             dt_total -= dt;

//             // Calculate wave propagation
//             calc_grid();
//         }

//         // Compute height of each vertex
//         adjust_grid();

//         // Draw wave grid to OpenGL display
//         draw_scene(window);

//         glfwPollEvents();
//     }

//     glfwTerminate();
//     exit(EXIT_SUCCESS);
// }

#include <iostream>

using namespace std;

int main()
{
    return 0;
}
