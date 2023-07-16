#include "mope_vec.hxx"
#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Testing works", "[testing]") {
    REQUIRE( 1 == 1 );
}

/*
#include "CppUnitTest.h"

#include "../mope_vec.hxx"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace mope;

template<> inline std::wstring
Microsoft::VisualStudio::CppUnitTestFramework::ToString<vec3i> (const vec3i& t) {
	RETURN_WIDE_STRING(t);
}
template<> inline std::wstring
Microsoft::VisualStudio::CppUnitTestFramework::ToString<vec3d>(const vec3d& t) {
	RETURN_WIDE_STRING(t);
}
template<> inline std::wstring
Microsoft::VisualStudio::CppUnitTestFramework::ToString<vec3f>(const vec3f& t) {
	RETURN_WIDE_STRING(t);
}
template<> inline std::wstring
Microsoft::VisualStudio::CppUnitTestFramework::ToString<mat3i>(const mat3i& t) {
	RETURN_WIDE_STRING(t);
}


namespace tests
{
	TEST_CLASS(ConstructorProperties)
	{
		template<template <class> class Property, class... Ty>
		void check_property()
		{
			(Assert::IsTrue(Property<Ty>::value), ...);
		}

		// Plagiarized from:
		// https://stackoverflow.com/questions/16893992/check-if-type-can-be-explicitly-converted
		// Someday I will understand it
		template<typename From, typename To>
		struct is_explicitly_convertible
		{
			template<typename T>
			static void f(T);

			template<typename F, typename T>
			static constexpr auto test(int) ->
				decltype(f(static_cast<T>(std::declval<F>())), true) {
				return true;
			}

			template<typename F, typename T>
			static constexpr auto test(...) -> bool {
				return false;
			}

			static bool const value = test<From, To>(0);
		};

	public:
		TEST_METHOD(DefaultConstructible)
		{
			check_property<std::is_default_constructible, vec3i, mat2i>();
		}

		TEST_METHOD(CopyConstuctible)
		{
			check_property<std::is_copy_constructible, vec3i, mat2i>();
		}

		TEST_METHOD(MoveConstuctible)
		{
			check_property<std::is_move_constructible, vec3i, mat2i>();
		}
		
		TEST_METHOD(CopyAssignable)
		{
			check_property<std::is_copy_assignable, vec3i, mat2i>();
		}

		TEST_METHOD(MoveAssignable)
		{
			check_property<std::is_move_assignable, vec3i, mat2i>();
		}

		TEST_METHOD(ImplicitlyConvertibleToHigherDimensions)
		{
			Assert::IsTrue(std::is_convertible_v<vec2i, vec3i>);
		}

		TEST_METHOD(NotImplicitlyConvertibleToLowerDimensions)
		{
			Assert::IsFalse(std::is_convertible_v<vec3i, vec2i>);
		}

		TEST_METHOD(NotExplicitlyConvertibleToLowerDimensions)
		{
			Assert::IsFalse(is_explicitly_convertible<vec3i, vec2i>::value);
		}

		TEST_METHOD(ExplicitlyConvertibleToOtherType)
		{
			Assert::IsTrue(is_explicitly_convertible<vec3i, vec3f>::value);
			Assert::IsTrue(is_explicitly_convertible<mat3i, mat3f>::value);
		}

		TEST_METHOD(NotImplicitlyConvertibleToOtherType)
		{
			Assert::IsFalse(std::is_convertible_v<vec3i, vec3f>);
			Assert::IsFalse(std::is_convertible_v<mat3i, mat3f>);
		}

		TEST_METHOD(ExplicitlyConvertibleToOtherTypeAndHigherDimension)
		{
			Assert::IsTrue(is_explicitly_convertible<vec3i, vec4f>::value);
		}
	};

	TEST_CLASS(VectorArithmetic)
	{
		vec3i u{ 0, 1, 2 };
		vec3i v{ 1, 2, 3 };

	public:
		TEST_METHOD(Addition)
		{
			vec3i expected{ 1, 3, 5 };

			Assert::AreEqual(expected, u + v);
			Assert::AreEqual(expected, v + u);
		}

		TEST_METHOD(Subtraction)
		{
			vec3i expected{ 1, 1, 1 };

			Assert::AreEqual(expected, v - u);
			Assert::AreEqual(-expected, u - v);
		}

		TEST_METHOD(ScalarMultiplication)
		{
			vec3i expected{ 0, 2, 4};

			Assert::AreEqual(expected, 2 * u);
			Assert::AreEqual(expected, u * 2);
		}

		TEST_METHOD(DotProduct)
		{
			int expected{ 8 };

			Assert::AreEqual(expected, u.dot(v));
			Assert::AreEqual(expected, v.dot(u));
		}

		TEST_METHOD(CrossProduct)
		{
			vec3i expected{ -1, 2, -1 };

			Assert::AreEqual(expected, u.cross(v));
		}

		TEST_METHOD(UnitVector)
		{
			vec3d actual1{ u.unit() };
			vec3f actual2{ u.unitf() };
			
			double root5 = sqrt(5);
			float root5f = sqrtf(5);
			vec3d expected1{ 0.0, 1.0 / root5, 2.0 / root5 };
			vec3f expected2{ 0.f, 1.f / root5f, 2.f / root5f };
			
			Assert::AreEqual(expected1, actual1);
			Assert::AreEqual(expected2, actual2);
		}
	};

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