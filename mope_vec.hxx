#pragma once

#include <type_traits>
#include <iostream>
#include <array>
#include <cmath>
#include <cstring>
#include <cassert>

namespace mope
{
    template <size_t N, typename T> struct vec;
    template <size_t N, typename V> struct mat;

    typedef vec<2, int>				vec2i;
    typedef vec<3, int>				vec3i;
    typedef vec<4, int>				vec4i;
    typedef vec<2, float>			vec2f;
    typedef vec<3, float>			vec3f;
    typedef vec<4, float>			vec4f;
    typedef vec<2, double>			vec2d;
    typedef vec<3, double>			vec3d;
    typedef vec<4, double>			vec4d;
    typedef vec<2, unsigned int>   	vec2ui;
    typedef vec<3, unsigned int>   	vec3ui;
    typedef vec<4, unsigned int>    vec4ui;
    typedef vec<2, uint8_t>			vec2b;
    typedef vec<3, uint8_t>			vec3b;
    typedef vec<4, uint8_t>			vec4b;

    typedef mat<2, vec2i>			mat2i;
    typedef mat<3, vec3i>			mat3i;
    typedef mat<4, vec4i>			mat4i;
    typedef mat<2, vec2f>			mat2f;
    typedef mat<3, vec3f>			mat3f;
    typedef mat<4, vec4f>			mat4f;
    typedef mat<2, vec2d>			mat2d;
    typedef mat<3, vec3d>			mat3d;
    typedef mat<4, vec4d>			mat4d;
    typedef mat<2, vec2ui>			mat2ui;
    typedef mat<3, vec3ui>			mat3ui;
    typedef mat<4, vec4ui>			mat4ui;

    namespace
    {
        /*====================================================================*\
        |  _base:                                                              |
        |  Contains operations common to both vectors and matrices             |
        |  (matrices are like row vectors over the field of column vectors)    |
        |                                                                      |
        |   N		The dimension (# of columns for a matrix)                  |
        |   T		The inner type (column vectors for a matrix)               |
        |   Vec     The derived type (see CRTP)                                |
        \*====================================================================*/
        template <size_t N, typename T, template <size_t, typename> class Vec>
        struct _base
        {
        protected:
            std::array<T, N> elements;

        public:
            constexpr _base() = default;
            constexpr _base(const std::initializer_list<T>& L) {
                assert(L.size() <= N);
                auto iter = L.begin( );
                auto arr = elements.begin( );
                for(
                    auto iter = L.begin( ), arr = elements.begin( );
                    iter < L.end( );
                    std::advance( arr, 1 ), std::advance( iter, 1 )
                )
                    std::copy( iter, std::next( iter ), arr );
            }

            // explicit cast to different data type
            template <typename S>
            constexpr explicit operator Vec<N, S>() const
            {
                Vec<N, S> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = static_cast<S>((*this)[i]);
                return res;
            }

            // subscript operator
            constexpr T& operator [] (const size_t& i)
            {
                return elements[i];
            }
            constexpr const T& operator [] (const size_t& i) const
            {
                return elements[i];
            }

            // element-wise addition
            template <typename S>
            constexpr auto operator + (const Vec<N, S>& rhs) const
            {
                Vec<N, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = (*this)[i] + rhs[i];
                return res;
            }

            // element-wise subtraction
            template <typename S>
            constexpr auto operator - (const Vec<N, S>& rhs) const
            {
                Vec<N, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = (*this)[i] - rhs[i];
                return res;
            }

            // negation
            constexpr auto operator - () const
            {
                Vec<N, T> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = -(*this)[i];
                return res;
            }

            // scalar multiplication
            template <typename S>
            constexpr auto operator * (S rhs) const
            {
                Vec<N, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = (*this)[i] * rhs;
                return res;
            }

            // assignments
            template <typename S>
            constexpr auto& operator += (const Vec<N, S>& rhs)
            {
                for (size_t i = 0; i < N; ++i)
                    (*this)[i] += rhs[i];
                return *this;
            }

            template <typename S>
            constexpr auto& operator += (const std::initializer_list<S>& L)
            {
                static_assert(L.size() <= N);
                size_t i = 0;
                for (auto iter = L.begin(); iter != L.end(); iter++)
                    (*this)[++i] += *iter;
                return *this;
            }

            template <typename S>
            constexpr auto& operator -= (const Vec<N, S>& rhs)
            {
                for (size_t i = 0; i < N; ++i)
                    (*this)[i] -= rhs[i];
                return *this;
            }

            template <typename S>
            constexpr auto& operator -= (const std::initializer_list<S>& L)
            {
                static_assert(L.size() <= N);
                size_t i = 0;
                for (auto iter = L.begin(); iter != L.end(); iter++)
                    (*this)[++i] -= *iter;
                return *this;
            }

            constexpr auto& operator *= (float rhs)
            {
                for (size_t i = 0; i < N; ++i)
                    (*this)[i] *= rhs;
                return *this;
            }

            // equality
            constexpr bool operator == (const Vec<N, T>& other) const
            {
                return this->elements == other.elements;
            }

            constexpr bool operator != (const Vec<N, T>& other) const
            {
                return this->elements != other.elements;
            }
        }; // struct _base

        // Left-multiplication by int scalar.
        template <size_t N, typename T, template <size_t, typename> class Vec>
        constexpr auto operator * (int lhs, const _base<N, T, Vec>& rhs)
        {
            return rhs * lhs;
        }

        // Left-multiplication by unsigned int scalar.
        template <size_t N, typename T, template <size_t, typename> class Vec>
        constexpr auto operator * (unsigned int lhs, const _base<N, T, Vec>& rhs)
        {
            return rhs * lhs;
        }

        // Left-multiplication by float scalar.
        template <size_t N, typename T, template <size_t, typename> class Vec>
        constexpr auto operator * (float lhs, const _base<N, T, Vec>& rhs)
        {
            return rhs * lhs;
        }
        // Left-multiplication by double scalar.
        template <size_t N, typename T, template <size_t, typename> class Vec>
        constexpr auto operator * (double lhs, const _base<N, T, Vec>& rhs)
        {
            return rhs * lhs;
        }

        template <size_t N, typename T>
        struct _vec : public _base<N, T, vec>
        {
            using _base<N, T, vec>::_base;

            // implicit cast to higher dimensions
            template <
                size_t P,
                typename PreventCastToLowerDimension = std::enable_if_t<P >= N>>
            constexpr operator vec<P, T>() const
            {
                vec<P, T> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = (*this)[i];
                for( size_t i = N; i < P;  ++i)
                    res[i] = 0;
                return res;
            }

            // explicit cast to different data type and higher dimension
            template <
                size_t P,
                typename S,
                typename PreventCastToLowerDimension = std::enable_if_t<P >= N>>
            constexpr explicit operator vec<P, S>() const
            {
                return static_cast<vec<P, S>>(static_cast<vec<N, S>>(*this));
            }

            // dot product
            template <typename S>
            constexpr auto dot(const _vec<N, S>& rhs) const
            {
                std::common_type_t<T, S> res = 0;
                for (size_t i = 0; i < N; ++i)
                    res += (*this)[i] * rhs[i];
                return res;
            }

            // cross product
            template <typename S>
            constexpr auto cross(const _vec<N, S>& rhs) const
            {
                static_assert(N > 2, "Cross product is not defined for n < 3.");

                vec<N, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < N - 2; ++i)
                    res[i] = (*this)[i + 1] * rhs[i + 2] - (*this)[i + 2] * rhs[i + 1];
                res[N - 2] = (*this)[N - 1] * rhs[0] - (*this)[0] * rhs[N - 1];
                res[N - 1] = (*this)[0] * rhs[1] - (*this)[1] * rhs[0];
                return res;
            }

            // elementwise scaling
            template <typename S>
            constexpr auto scaleby(const _vec<N, S>& v) const
            {
                vec< N, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < N; ++i)
                    res[i] = (*this)[i] * v[i];
                return res;
            }

            // magnitude
            double magnitude() const
            {
                double d = static_cast<double>(dot(*this));
                return std::sqrt(d);
            }

            // magnitude as a float
            float magnitudef() const
            {
                float d = static_cast<float>(dot(*this));
                return std::sqrt(d);
            }

            // unit vector
            vec<N, double> unit() const
            {
                double factor = 1.0 / this->magnitude();
                return (*this) * factor;
            }

            // unit vector as floats
            vec<N, float> unitf() const
            {
                float factor = 1.f / this->magnitudef();
                return (*this) * factor;
            }
        }; // struct _vec

        template <size_t M, size_t N, typename T>
        struct _mat : public _base<N, vec<M, T>, mat>
        {
            using _base<N, vec<M, T>, mat>::_base;

            constexpr _mat(const std::initializer_list<T>& L)
            {
                assert( L.size( ) <= M * N );
                auto iter = L.begin( );
                size_t idx = 0;
                for( size_t idx ; iter < L.end( ); ++idx )
                {
                    auto beg = iter;
                    std::advance( iter, M );
                    auto end = L.end( ) < iter ? L.end( ) : iter;
                    std::copy( beg, end, ( *this )[idx].data( ) );
                }
            }

            // matrix - vector multiplication
            template <typename S>
            constexpr auto operator * (const vec<N, S>& rhs) const
            {
                vec<M, std::common_type_t<T, S>> res;
                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < N; ++j)
                        res[i] += (*this)[j][i] * rhs[j];
                return res;
            }

            // matrix - matrix multiplication
            template <size_t P, typename S>
            constexpr auto operator * (const mat<P, vec<N, S>>& rhs) const
            {
                mat<P, vec<M, std::common_type_t<T, S>>> res;
                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < N; j++)
                        for (size_t k = 0; k < P; k++)
                            res[k][i] += (*this)[j][i] * rhs[k][j];
                return res;
            }

            using _base<N, vec<M, T>, mat>::operator*;

            // transpose
            constexpr auto transpose() const
            {
                mat<M, vec<N, T>> res;
                for (size_t i = 0; i < M; ++i)
                    for (size_t j = 0; j < N; ++j)
                        res[i][j] = (*this)[j][i];
                return res;
            }

            // get all the column-ordered elements as a vector
            constexpr auto unpack() const
            {
                std::array<T, M* N> dat;
                for (size_t j = 0; j < N; ++j)
                    for (size_t i = 0; i < M; ++i)
                        dat[j * N + i] = (*this)[j][i];
                return dat;
            }

            // special features square matrices
            using square_mat = std::enable_if_t<M == N, mat<N, vec<N, T>>>;

            constexpr square_mat inverse()
            {

            }

            constexpr static square_mat identity()
            {
                mat<N, vec<N, T>> res{};
                for (size_t i = 0; i < N; ++i)
                    res[i][i] = static_cast<T>(1);
                return res;
            }
        }; // struct _mat
    } // anonymous namespace

    /*========================================================================*\
    |  vec                                                                     |
    \*========================================================================*/

    // N-dimensional vector
    template <size_t N, typename T>
    struct vec : public _vec<N, T>
    {
        using _vec<N, T>::_vec;
    };

    // 2-dimensional vector
    template <class T>
    struct vec<2, T> : public _vec<2, T>
    {
        using _vec<2, T>::_vec;

        constexpr T& x() { return (*this)[0]; }
        constexpr T& y() { return (*this)[1]; }
        constexpr const T& x() const { return (*this)[0]; }
        constexpr const T& y() const { return (*this)[1]; }
    };

    // 3-dimensional vector
    template <class T>
    struct vec<3, T> : public _vec<3, T>
    {
        using _vec<3, T>::_vec;

        constexpr T& x() { return (*this)[0]; }
        constexpr T& y() { return (*this)[1]; }
        constexpr T& z() { return (*this)[2]; }
        constexpr const T& x() const { return (*this)[0]; }
        constexpr const T& y() const { return (*this)[1]; }
        constexpr const T& z() const { return (*this)[2]; }
    };

    // 4-dimensional vector
    template <class T>
    struct vec<4, T> : public _vec<4, T>
    {
        using _vec<4, T>::_vec;

        constexpr T& x() { return (*this)[0]; }
        constexpr T& y() { return (*this)[1]; }
        constexpr T& z() { return (*this)[2]; }
        constexpr T& w() { return (*this)[3]; }
        constexpr const T& x() const { return (*this)[0]; }
        constexpr const T& y() const { return (*this)[1]; }
        constexpr const T& z() const { return (*this)[2]; }
        constexpr const T& w() const { return (*this)[3]; }
    };


    /*========================================================================*\
    |  mat                                                                     |
    \*========================================================================*/

    // Column-major matrix class
    template <size_t M, size_t N, typename T>
    struct mat<N, vec<M, T>> : public _mat<M, N, T>
    {
        using _mat<M, N, T>::_mat;
    };

    // partial specialization allows:
    // mat<2, ve<2, int>>
    // absense of other partial specializations prevents things like:
    // mat<2, int> m;

    /*========================================================================*\
    |  Conveniences                                                            |
    \*========================================================================*/

    // Stream printing
    template <size_t N, typename T, template <size_t, typename> class Vec>
    std::ostream& operator<<(std::ostream& os, const _base<N, T, Vec>& v) {
        os << "( ";
        for (size_t i = 0; i < N; ++i) {
            os << v[i] << " ";
        }
        os << ")";
        return os;
    }
    // Stream printing (wide)
    template <size_t N, typename T, template <size_t, typename> class Vec>
    std::wostream& operator<<(std::wostream& os, const _base<N, T, Vec>& v) {
        os << "( ";
        for (size_t i = 0; i < N; ++i) {
            os << v[i] << " ";
        }
        os << ")";
        return os;
    }
} // namespace mope

// common type of two vecs with different inner data types
template <size_t N, typename T1, typename T2>
struct std::common_type<mope::vec<N, T1>, mope::vec<N, T2>> {
    using type = mope::vec<N, std::common_type_t<T1, T2>>;
};
// common type of a vec and a different data type
template <size_t N, typename T1, typename T2>
struct std::common_type<mope::vec<N, T1>, T2> {
    using type = mope::vec<N, std::common_type_t<T1, T2>>;
};
// common type of a different data type and a vec
template <size_t N, typename T1, typename T2>
struct std::common_type<T1, mope::vec<N, T2>> {
    using type = mope::vec<N, std::common_type_t<T1, T2>>;
};