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
        constexpr mope::vec2i expected0{ 1, 3 };
        constexpr mope::vec2i expected1{ 2, 4 };
        auto dat = A.data( );
        REQUIRE( dat[0] == expected0 );
        REQUIRE( dat[1] == expected1 );
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
        constexpr mope::mat2i B{
            { 1, 3 },
            { 2, 4 }
        };
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

TEST_CASE( "vector arithmetic works as expected", "[vector arithmetic]" ) {
    constexpr mope::vec3i x{ 0, 1, 2 };
    constexpr mope::vec3i y{ 1, 2, 3 };
	mope::vec3i w = x;

    SECTION( "addition is correct" ) {
		constexpr mope::vec3i expected{ 1, 3, 5 };
		REQUIRE(x + y == expected);
		REQUIRE(y + x == expected);
	}

	SECTION( "addition-assignment is correct" ) {
        constexpr mope::vec3i expected{ 1, 3, 5 };
        w += y;
        REQUIRE( w == expected );
    }

    SECTION( "subtraction is correct" ) {
		constexpr mope::vec3i expected0{ 1, 1, 1 };
		constexpr mope::vec3i expected1{ -1, -1, -1 };
		REQUIRE(y - x == expected0);
		REQUIRE(x - y == expected1);
	}

	SECTION( "subtraction-assignment is correct" ) {
        constexpr mope::vec3i expected{ -1, -1, -1 };
        w -= y;
        REQUIRE( w == expected );
    }
	
	SECTION( "negation is correct" ) {
		constexpr mope::vec3i expected{ -1, -2, -3 };
		REQUIRE(-y == expected);
	}

	SECTION( "scalar multiplication is correct" ) {
		constexpr mope::vec3i expected{ 0, 2, 4};
		REQUIRE(2 * x == expected);
		REQUIRE(x * 2 == expected);
	}

	SECTION( "dot product is correct" ) {
		constexpr int expected{ 8 };
		REQUIRE(x.dot(y) == expected);
		REQUIRE(y.dot(x) == expected);
	}

	SECTION( "cross product is correct" ) {
		constexpr mope::vec3i expected{ -1, 2, -1 };
		REQUIRE(x.cross(y) == expected);
	}

	SECTION( "unit vector is correct" ) {	
		const mope::vec3d expected{ 0.0, 1.0 / std::sqrt(5), 2.0 / std::sqrt(5) };
		const mope::vec3f expectedf{ 0.f, 1.f / std::sqrt(5.f), 2.f / std::sqrt(5.f) };
		REQUIRE(x.unit() == expected);
		REQUIRE(x.unitf() == expectedf);
	}
}

/*


	TEST_CLASS(MatrixArithmetic)
	{
		mat3i m{
			0, 1, 2,
			3, 4, 5,
			6, 7, 8
		};

		mat3i n{
			6, 7, 8,
			3, 4, 5,
			0, 1, 2
		};

		TEST_METHOD(Addition)
		{
			mat3i expected{
				6, 8, 10,
				6, 8, 10,
				6, 8, 10
			};

			Assert::AreEqual(expected, m + n);
			Assert::AreEqual(expected, n + m);
		}

		TEST_METHOD(Subtraction)
		{
			mat3i expected{
				6, 6, 6,
				0, 0, 0,
				-6, -6, -6
			};

			Assert::AreEqual(expected, n - m);
			Assert::AreEqual(-expected, m - n);
		}

		TEST_METHOD(MatrixMatrixMultiplication)
		{
			mat3i expected{
				69, 90, 111,
				42, 54, 66,
				15, 18, 21
			};

			Assert::AreEqual(expected, m * n);
		}

		TEST_METHOD(MatrixVectorMultiplication)
		{
			vec3i v{ 1, 2, 3 };
			vec3i expected{
				24, 30, 36
			};

			Assert::AreEqual(expected, m * v);
		}

		TEST_METHOD(Identity)
		{
			mat3i I = mat3i::identity();

			Assert::AreEqual(m, m * I);
			Assert::AreEqual(m, I * m);
		}

		TEST_METHOD(Transpose)
		{
			mat3i expected = {
				0, 3, 6,
				1, 4, 7,
				2, 5, 8
			};

			Assert::AreEqual(expected, m.transpose());
		}
	};
}

*/