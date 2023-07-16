#include "mope_vec.hxx"
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

TEST_CASE( "vecs and mats have desired type traits", "[type traits]") {
	SECTION( "vecs and mats are default constructible" ) {
		REQUIRE( std::is_default_constructible_v<mope::vec2i> );
		REQUIRE( std::is_default_constructible_v<mope::mat2i> );
	}

	SECTION( "vecs and mats are copy constructible" ) {
		REQUIRE( std::is_copy_constructible_v<mope::vec2i> );
		REQUIRE( std::is_copy_constructible_v<mope::mat2i> );
	}

	SECTION( "vecs and mats copy assignable" ) {
		REQUIRE( std::is_copy_assignable_v<mope::vec2i> );
		REQUIRE( std::is_copy_assignable_v<mope::mat2i> );
	}

	SECTION( "vecs and mats move constructible" ) {
		REQUIRE( std::is_move_constructible_v<mope::vec2i> );
		REQUIRE( std::is_move_constructible_v<mope::mat2i> );
	}

	SECTION( "vecs and mats move assignable" ) {
		REQUIRE( std::is_move_assignable_v<mope::vec2i> );
		REQUIRE( std::is_move_assignable_v<mope::mat2i> );
	}

	SECTION( "vecs are convertible from lower dimensions" ) {
		REQUIRE( std::is_convertible_v<mope::vec2i, mope::vec3i> );
	}

	SECTION( "vecs are not convertible from higher dimensions" ) {
		REQUIRE_FALSE( std::is_constructible_v<mope::vec2i, mope::vec3i> );
	}

	SECTION( "vecs and mats are explicitly convertible to another field type" ) {
		REQUIRE_FALSE( std::is_convertible_v<mope::vec2i, mope::vec2f> );
		REQUIRE( std::is_constructible_v<mope::vec2f, mope::vec2i> );
	}
}

TEST_CASE( "preliminary conditions for testing are met", "[preliminary]") {
	constexpr mope::vec2i x{ 1, 2 };
	constexpr mope::mat2i A{ 1, 3, 2, 4 };

	SECTION( "basic vec construction works as expected" ) {
		auto dat = x.data( );
		REQUIRE( dat[0] == 1 );
		REQUIRE( dat[1] == 2 );
	}

    SECTION( "vec equality and inequality work as expected" ) {
		constexpr mope::vec2i y{ 1, 2 };
        constexpr mope::vec2i z1{ 2, 2 };
        constexpr mope::vec2i z2{ 1, 1 };
        REQUIRE( x == y );
		REQUIRE_FALSE( x == z1 );
		REQUIRE_FALSE( x == z2 );
		REQUIRE( x != z1 );
		REQUIRE( x != z2 );
		REQUIRE_FALSE( x != y );
	}

	SECTION( "basic mat construction works as expected" ) {
        constexpr mope::vec2i expected_0{ 1, 3 };
        constexpr mope::vec2i expected_1{ 2, 4 };
        auto dat = A.data( );
        REQUIRE( dat[0] == expected_0 );
        REQUIRE( dat[1] == expected_1 );
    }

	SECTION( "mat equality and inequality work as expected" ) {
		constexpr mope::mat2i B{ 1, 3, 2, 4 };
        constexpr mope::mat2i C1{ 1, 3, 1, 3 };
        constexpr mope::mat2i C2{ 2, 4, 2, 4 };
        REQUIRE( A == B );
		REQUIRE_FALSE( A == C1 );
		REQUIRE_FALSE( A == C2 );
		REQUIRE( A != C1 );
		REQUIRE( A != C2 );
		REQUIRE_FALSE( A != B );
	}

	SECTION( "packed and unpacked mat construction give the same results" ) {
        constexpr mope::mat2i B{ { 1, 3 }, { 2, 4 } };
        REQUIRE( A == B );
    }

    SECTION( "copying and moving work as expected") {
		constexpr mope::vec2i x_expected{ 1, 2 };
		constexpr mope::mat2i A_expected{ 1, 3, 2, 4 };

		SECTION( "copying construction works as expected" ) {
			constexpr mope::vec2i x_copy( x );
			constexpr mope::mat2i A_copy( A );
			REQUIRE( x == x_expected );
			REQUIRE( A == A_expected );
			REQUIRE( x == x_copy );
			REQUIRE( A == A_copy );
		}

		SECTION( "copy assignment works as expected" ) {
			constexpr mope::vec2i x_copy = x;
			constexpr mope::mat2i A_copy = A;
			REQUIRE( x == x_expected );
			REQUIRE( A == A_expected );
			REQUIRE( x == x_copy );
			REQUIRE( A == A_copy );
		}

		SECTION( "move construction works as expected" ) {
			constexpr mope::vec2i x_move = std::move( x );
			constexpr mope::mat2i A_move = std::move( A );
			REQUIRE( x_move == x_expected );
			REQUIRE( A_move == A_expected );
		}

		SECTION( "move assignment works as expected" ) {
			constexpr mope::vec2i x_move{ std::move( x ) };
			constexpr mope::mat2i A_move{ std::move( A ) };
			REQUIRE( x_move == x_expected );
			REQUIRE( A_move == A_expected );
		}
	}
}

TEST_CASE( "common operations work as expected", "[common operations]" ) {
	constexpr mope::vec3i x{ 0, 1, 2 };
    constexpr mope::vec3i y{ 1, 2, 3 };
	constexpr mope::mat3i A{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	constexpr mope::mat3i B{ 6, 7, 8, 3, 4, 5, 0, 1, 2 };
    mope::vec3i x_copy{ x };
    mope::mat3i A_copy{ A };

    SECTION( "addition works as expected" ) {
    	constexpr mope::vec3i x_plus_y{ 1, 3, 5 };
		constexpr mope::mat3i A_plus_B{ 6, 8, 10, 6, 8, 10, 6, 8, 10 };

		SECTION( "sum is correct" ) {
            REQUIRE( x + y == x_plus_y );
            REQUIRE( y + x == x_plus_y );
            REQUIRE( A + B == A_plus_B );
            REQUIRE( B + A == A_plus_B );
        }

        SECTION( "addition-assignment is correct" ) {
			x_copy += y;
            REQUIRE( x_copy == x_plus_y );
            A_copy += B;
            REQUIRE( A_copy == A_plus_B );
        }
    }

	SECTION( "subtraction works as expected" ) {
    	constexpr mope::vec3i x_minus_y{ -1, -1, -1 };
		constexpr mope::vec3i y_minus_x{ 1, 1, 1 };
	    constexpr mope::vec3i minus_x{ 0, -1, -2 };
        constexpr mope::mat3i B_minus_A{ 6, 6, 6, 0, 0, 0, -6, -6, -6 };
        constexpr mope::mat3i A_minus_B{ -6, -6, -6, 0, 0, 0, 6, 6, 6 };
        constexpr mope::mat3i minus_A{ 0, -1, -2, -3, -4, -5, -6, -7, -8 };

        SECTION( "difference is correct" ) {
            REQUIRE( x - y == x_minus_y );
            REQUIRE( y - x == y_minus_x );
            REQUIRE( A - B == A_minus_B );
            REQUIRE( B - A == B_minus_A );
        }

        SECTION( "subtraction-assignment is correct" ) {
			x_copy -= y;
            REQUIRE( x_copy == x_minus_y );
            A_copy -= B;
            REQUIRE( A_copy == A_minus_B );
        }

        SECTION( "negation is correct" ) {
            REQUIRE( -x == minus_x );
            REQUIRE( -A == minus_A );
        }
    }

	SECTION( "scalar multiplication works as expected" ) {
		constexpr mope::vec3i two_times_x{ 0, 2, 4};
		constexpr mope::mat3i two_times_A{ 0, 2, 4, 6, 8, 10, 12, 14, 16 };

		SECTION( "product is correct" ) {
            REQUIRE( 2 * x == two_times_x );
            REQUIRE( x * 2 == two_times_x );
            REQUIRE( 2 * A == two_times_A );
            REQUIRE( A * 2 == two_times_A );
        }

        SECTION( "scalar-multiplication-assignment is correct" ) {
            x_copy *= 2;
            REQUIRE( x_copy == two_times_x );
            A_copy *= 2;
            REQUIRE( A_copy == two_times_A );
        }
	}
}

TEST_CASE( "vector operations work as expected", "[vector operations]" ) {
    constexpr mope::vec3i x{ 0, 1, 2 };
    constexpr mope::vec3i y{ 1, 2, 3 };

	SECTION( "dot product is correct" ) {
		constexpr int x_dot_y{ 8 };
		REQUIRE( x.dot(y) == x_dot_y );
		REQUIRE( y.dot(x) == x_dot_y );
	}

	SECTION( "cross product is correct" ) {
		constexpr mope::vec3i x_cross_y{ -1, 2, -1 };
		REQUIRE(x.cross(y) == x_cross_y);
	}

	SECTION( "magnitude is correct" ) {
        const double x_magnitude = std::sqrt( 5 );
        const float x_magnitudef = std::sqrt( 5.f );
        REQUIRE( x.magnitude() == x_magnitude );
        REQUIRE( x.magnitudef() == x_magnitudef );
    }

    SECTION( "unit vector is correct" ) {
        const mope::vec3d x_unit{ 0.0, 1.0 / std::sqrt( 5 ), 2.0 / std::sqrt( 5 ) };
        const mope::vec3f x_unitf{ 0.f, 1.f / std::sqrt( 5.f ), 2.f / std::sqrt( 5.f ) };
        REQUIRE(x.unit() == x_unit);
		REQUIRE(x.unitf() == x_unitf);
	}
}

TEST_CASE( "matrix operations work as expected", "[matrix operations]") {
	constexpr mope::mat3i A{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
	constexpr mope::mat3i B{ 6, 7, 8, 3, 4, 5, 0, 1, 2 };

	SECTION( "matrix-matrix multiplication is correct" ) {
        constexpr mope::mat3i A_times_B{ 69, 90, 111, 42, 54, 66, 15, 18, 21 };
        REQUIRE( A * B == A_times_B );
    }

	SECTION( "matrix-vector multiplication is correct" ) {
		constexpr mope::vec3i x{ 1, 2, 3 };
		constexpr mope::vec3i A_times_x{ 24, 30, 36 };
        REQUIRE( A * x == A_times_x );
    }

    SECTION( "identity behaves as expected" ) {
        constexpr mope::mat3i I = mope::mat3i::identity( );
		REQUIRE( A * I == A);
		REQUIRE( I * A == A);
    }

	SECTION( "transpose is correct" ) {
        constexpr mope::mat3i A_transpose{ 0, 3, 6, 1, 4, 7, 2, 5, 8 };
        REQUIRE( A.transpose( ) == A_transpose );
    }
}
