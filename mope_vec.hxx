#pragma once

#include <type_traits>
#include <iostream>
#include <array>
#include <vector>
#include <functional>
#include <cmath>
#include <cstring>


namespace mope
{
    template <size_t N, typename T> struct vec;
    template <size_t N, typename V> struct mat;

    typedef vec<2, int>				    vec2i;
    typedef vec<3, int>				    vec3i;
    typedef vec<4, int>				    vec4i;
    typedef vec<2, float>			    vec2f;
    typedef vec<3, float>			    vec3f;
    typedef vec<4, float>			    vec4f;
    typedef vec<2, double>			    vec2d;
    typedef vec<3, double>			    vec3d;
    typedef vec<4, double>			    vec4d;
    typedef vec<2, unsigned int>        vec2ui;
    typedef vec<3, unsigned int>        vec3ui;
    typedef vec<4, unsigned int>        vec4ui;
    typedef vec<2, uint8_t>			    vec2b;
    typedef vec<3, uint8_t>			    vec3b;
    typedef vec<4, uint8_t>			    vec4b;

    typedef mat<2, vec2i>			    mat2i;
    typedef mat<3, vec3i>			    mat3i;
    typedef mat<4, vec4i>			    mat4i;
    typedef mat<2, vec2f>			    mat2f;
    typedef mat<3, vec3f>			    mat3f;
    typedef mat<4, vec4f>			    mat4f;
    typedef mat<2, vec2d>			    mat2d;
    typedef mat<3, vec3d>			    mat3d;
    typedef mat<4, vec4d>			    mat4d;
    typedef mat<2, vec2ui>			    mat2ui;
    typedef mat<3, vec3ui>			    mat3ui;
    typedef mat<4, vec4ui>			    mat4ui;

    namespace detail
    {

        template <typename T>
        struct is_field_helper
            : public std::conditional_t<std::is_floating_point_v<T>, std::true_type, std::false_type>
        { };

        // coming soon
        // struct fraction;
        // template <>
        // struct is_field_helper<fraction>
        //     : public std::true_type
        // { };
    } // namespace detail

    template <typename T>
    struct is_field
        : public detail::is_field_helper<std::remove_cv_t<T>>
    { };

    template <typename T>
    inline constexpr bool is_field_v = is_field<T>::value;

    namespace detail
    {
        // this non-constexpr function is called from constructors in order
        // to trigger a compiler error
        void too_many_elements( ){ }

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
            T elements[N]{ };

        public:
            constexpr _base() = default;
            constexpr _base(const std::initializer_list<T>& L)
            {
                L.size() <= N ? (void)0 : too_many_elements();
                std::copy( L.begin( ), L.end( ), elements );
            }

            // access to underlying array
            constexpr T* data() {
                return elements;
            }

            constexpr const T* data() const {
                return elements;
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
                L.size() <= N ? (void)0 : too_many_elements();
                *this += Vec<N, S>{ L };
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
                L.size() <= N ? (void)0 : too_many_elements();
                *this -= Vec<N, S>{ L };
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
                return !std::memcmp( elements, other.elements, N * sizeof(T) );
            }

            constexpr bool operator != (const Vec<N, T>& other) const
            {
                return !( ( *this ) == other );
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
            using Col = vec<M, T>;
            using _base<N, Col, mat>::_base;

            constexpr _mat(const std::initializer_list<T>& L)
            {
                L.size() <= M * N ? (void)0 : too_many_elements( );
                auto iter = L.begin( );
                auto remaining = std::distance( iter, L.end( ) );
                for( size_t idx = 0; remaining > 0; ++idx ) {
                    auto dist = M < remaining ? M : remaining;
                    std::copy( iter, std::next( iter, dist ), ( *this )[idx].data( ) );
                    std::advance( iter, dist );
                    remaining -= dist;
                }
            }

            using _base<N, Col, mat>::operator*;

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
                    for (size_t j = 0; j < N; ++j)
                        for (size_t k = 0; k < P; ++k)
                            res[k][i] += (*this)[j][i] * rhs[k][j];
                return res;
            }

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

            // these might come in handy
            using row_op = std::function<void( Col& )>;

            static row_op exchange(size_t row_1, size_t row_2)
            {
                return [row_1, row_2]( Col& a ) {
                    T tmp = a[row_1];
                    a[row_1] = a[row_2];
                    a[row_2] = tmp;
                };
            }

            static row_op scale(size_t row, T scalar)
            {
                return [row, scalar]( Col& a ) {
                    a[row] *= scalar;
                };
            }

            static row_op add_row(size_t this_row, size_t other_row, T scalar)
            {
                return [this_row, other_row, scalar]( Col& a ) {
                    a[this_row] += scalar * a[other_row];
                };
            }

            // destructively LUP-decomposes this matrix. Entries below the main
            // diagonal are part of L, and the main diagonal and above are part
            // of U, with the maindiagonal of L understood to all be 1. In short
            // converts this matrix to L + U - I, returning a vector of pivots
            // representing the filled columns in the permutation matrix.
            constexpr vec<N, size_t> LUPdecompose()
            {
                static_assert( is_field_v<T> && M == N,
                               "This method only makes sense on square matrices"
                               "over fields." );

                std::vector<row_op> row_ops;
                vec<N, size_t> pivots;
                for( size_t i = 0;  i < N; ++i )
                    pivots[i] = i;

                auto partial_pivot = [&row_ops, &pivots]( Col& a_i, size_t pivot_idx ) {
                    for ( size_t row = pivot_idx + 1; row < N; ++row ) {
                        if( a_i[row] != 0 ) {
                            row_ops.push_back( exchange( pivot_idx, row ) );
                            row_ops.back( )( a_i );
                            size_t tmp = pivots[row];
                            pivots[row] = pivots[pivot_idx];
                            pivots[pivot_idx] = tmp;
                            break;
                        }
                    }
                };

                auto add_rows = [&row_ops]( Col& a_i, size_t pivot_idx ) {
                    T denom = 1 / a_i[pivot_idx];
                    for( size_t row = pivot_idx + 1; row < N; ++row ) {
                        T scalar = a_i[row] * denom;
                        row_ops.push_back( add_row( row, pivot_idx, -scalar ) );
                        a_i[row] = scalar;
                    }
                };

                for (size_t i = 0; i < N; ++i)
                {
                    Col& col = ( *this )[i];
                    for( auto op : row_ops ) {
                        op( col );
                    }
                    if ( col[i] == 0 ) {
                        partial_pivot( col, i );
                        if (col[i] == 0 )
                            continue;
                    }
                    add_rows( col, i );
                }

                return pivots;
            }

            // destructively converts this matrix to its inverse, if one exists
            constexpr void invert( )
            {
                // inverse matrices probably don't make sense unless our field has
                // inverses. specialize is_field if you disagree.
                static_assert( is_field_v<T> && M == N,
                               "This method only makes sense on square matrices"
                               "over fields." );

                auto copy = *this;
                auto pivots = copy.LUPdecompose( );

                for( size_t col = 0; col < N; ++col )
                {
                    Col ycol{ };
                    ycol[pivots[col]] = 1;
                    for( size_t row = pivots[col] + 1; row < N; ++row )
                    {
                        T sum = 0;
                        for( size_t i = pivots[col]; i < row; ++i )
                        {
                            sum += copy[i][row] * ycol[i];
                        }
                        ycol[row] = -sum;
                    }

                    Col& xcol = ( *this )[col];
                    for( size_t row = N; row-- > 0; )
                    {
                        T sum = 0;
                        for( size_t i = N - 1; i > row; --i) {
                            sum += copy[i][row] * xcol[i];
                        }
                        // avoid signed zero
                        if (sum == ycol[row]) {
                            xcol[row] = 0;
                        }
                        else {
                            xcol[row] = ( ycol[row] - sum ) / copy[row][row];
                        }
                    }
                }
            }

            // return the inverse of this matrix
            constexpr mat<N, Col> inverse( ) const
            {
                static_assert( is_field_v<T> && M == N,
                               "This method only makes sense on square matrices"
                               "over fields." );

                mat<N, Col> cpy;
                std::copy( this->data( ), std::next( this->data( ), N ), cpy.data( ) );
                cpy.invert( );
                return cpy;
            }

            constexpr static mat<N, Col> identity( )
            {
                static_assert( M == N,
                               "identity generally only exists for square matrices" );
                mat<N, vec<N, T>> res{ };
                for (size_t i = 0; i < N; ++i)
                    res[i][i] = static_cast<T>(1);
                return res;
            }
        }; // struct _mat
    } // namespace detail

    /*========================================================================*\
    |  vec                                                                     |
    \*========================================================================*/

    // N-dimensional vector
    template <size_t N, typename T>
    struct vec : public detail::_vec<N, T>
    {
        using detail::_vec<N, T>::_vec;
    };

    // 2-dimensional vector
    template <class T>
    struct vec<2, T> : public detail::_vec<2, T>
    {
        using detail::_vec<2, T>::_vec;

        constexpr T& x() { return (*this)[0]; }
        constexpr T& y() { return (*this)[1]; }
        constexpr const T& x() const { return (*this)[0]; }
        constexpr const T& y() const { return (*this)[1]; }
    };

    // 3-dimensional vector
    template <class T>
    struct vec<3, T> : public detail::_vec<3, T>
    {
        using detail::_vec<3, T>::_vec;

        constexpr T& x() { return (*this)[0]; }
        constexpr T& y() { return (*this)[1]; }
        constexpr T& z() { return (*this)[2]; }
        constexpr const T& x() const { return (*this)[0]; }
        constexpr const T& y() const { return (*this)[1]; }
        constexpr const T& z() const { return (*this)[2]; }
    };

    // 4-dimensional vector
    template <class T>
    struct vec<4, T> : public detail::_vec<4, T>
    {
        using detail::_vec<4, T>::_vec;

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
    struct mat<N, vec<M, T>> : public detail::_mat<M, N, T>
    {
        using detail::_mat<M, N, T>::_mat;
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
    std::ostream& operator<<(std::ostream& os, const detail::_base<N, T, Vec>& v) {
        os << "( ";
        for (size_t i = 0; i < N; ++i) {
            os << v[i] << " ";
        }
        os << ")";
        return os;
    }
    // Stream printing (wide)
    template <size_t N, typename T, template <size_t, typename> class Vec>
    std::wostream& operator<<(std::wostream& os, const detail::_base<N, T, Vec>& v) {
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