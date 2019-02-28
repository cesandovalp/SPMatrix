#include <iostream>
#include <assert.h>
//#include "Matrix.hpp"
#include "SerialMatrixOperations.hpp"
#include "MultiCoreMatrixOperations.hpp"

using sp::Matrix;

void PrintMatrix( Matrix<int>& M )
{
  for( unsigned i = 0; i < M.rows; ++i )
  {
    for( unsigned j = 0; j < M.columns; ++j )
      std::cout << M[i][j] << "\t|";
    std::cout << std::endl;
  }
}

void FillMatrix( Matrix<int>& M )
{
  for( unsigned i = 0; i < M.rows; ++i )
    for( unsigned j = 0; j < M.columns; ++j )
      M.data[i * M.columns + j] = i * M.columns + j;
}

void TestAddition()
{
  Matrix<int> A( 4, 5 );
  Matrix<int> B( 4, 5 );

  sp::SetSerial( &A ); //SetSerial
  sp::SetSerial( &B );

  FillMatrix( A );
  FillMatrix( B );

  // Trace the instructions
  auto C = A + B;
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationVector()
{
  Matrix<int>      A( 2, 3 );
  std::vector<int> B( { 1, 2, 3 } );

  sp::SetSerial( &A ); //SetSerial

  FillMatrix( A );

  auto C = A * B;
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  std::cout << "| 1 |\n| 2 |\n| 3 |" << std::endl << std::endl;
  PrintMatrix( C );
}

void TestMultiplication()
{
  Matrix<int> A( 2, 3 );
  Matrix<int> B( 3, 4 );

  sp::SetSerial( &A );
  sp::SetSerial( &B );

  FillMatrix( A );
  FillMatrix( B );

  auto C = A * B;
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationNoResize()
{
  Matrix<int> A( 3, 3 );
  Matrix<int> B( 3, 3 );

  sp::SetSerial( &A );
  sp::SetSerial( &B );

  FillMatrix( A );
  FillMatrix( B );

  A.Multiplication( B );
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
}

void TestHadamardProduct()
{
  Matrix<int> A( 10, 10 );
  Matrix<int> B( 10, 10 );

  sp::SetSerial( &A );
  sp::SetSerial( &B );

  FillMatrix( A );
  FillMatrix( B );

  A.HadamardProduct( B );
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
}

void TestMultiplicationIn()
{
  Matrix<int> A( 3, 3 );
  Matrix<int> B( 3, 3 );
  Matrix<int> C( 3, 3 );

  sp::SetSerial( &A );
  sp::SetSerial( &B );
  sp::SetSerial( &C );

  FillMatrix( A );
  FillMatrix( B );

  C.Multiplication( A, B );
  //std::cout << "C memory address: " << &C << std::endl;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationScalar()
{
  Matrix<int> A( 2, 3 );

  sp::SetSerial( &A );

  FillMatrix( A );

  auto B = A;
  auto C = B * 2;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestTranspose()
{
  Matrix<int> A( 2, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  sp::SetSerial( &A );

  auto B = !A;
  PrintMatrix( B );
}

void TestMinus()
{
  Matrix<int> A( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  sp::SetSerial( &A );

  auto B = -A;
  PrintMatrix( B );
}

void TestCopy()
{
  Matrix<int> A( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  Matrix<int> B = A;
  PrintMatrix( B );
}

void TestApply()
{
  Matrix<int> A( 3, 3);
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  sp::SetSerial( &A );

  std::cout << "square" << std::endl;
  auto B = A.Apply( []( int x ) { return x * x; } );
  PrintMatrix( B );
  std::cout << "times 2" << std::endl;
  B = A.Apply( []( int x ) { return 2 * x; } );
  PrintMatrix( B );
  std::cout << "threshold 5" << std::endl;
  B = A.Apply( []( int x ) { return x < 5 ? 0 : x; } );
  PrintMatrix( B );
  std::cout << "transpose threshold 5" << std::endl;
  auto C = !B;
  PrintMatrix( C );
}

void TestToTarget()
{
  Matrix<int> A( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  std::cout << "To vector" << std::endl;
  auto B = A.ToVector();
  std::cout << "B.size() = " << B.size() << std::endl;
  for( unsigned i = 0; i < B.size(); ++i )
    std::cout << B[i] << std::endl;
}

int main()
{
  std::cout << "################## Test Addition ################" << std::endl;
  TestAddition();
  std::cout << "############### Test Multiplication #############" << std::endl;
  TestMultiplication();
  std::cout << "############ Test Multiplication Same ###########" << std::endl;
  TestMultiplicationNoResize();
  std::cout << "############# Test Hadamard Product #############" << std::endl;
  TestHadamardProduct();
  std::cout << "########## Test Multiplication Vector ###########" << std::endl;
  TestMultiplicationVector();
  std::cout << "########### Test Multiplication Scalar ##########" << std::endl;
  TestMultiplicationScalar();
  std::cout << "############# Test Multiplication In ############" << std::endl;
  TestMultiplicationIn();
  std::cout << "################# Test Transpose ################" << std::endl;
  TestTranspose();
  std::cout << "################### Test Minus ##################" << std::endl;
  TestMinus();
  std::cout << "################### Test Copy ###################" << std::endl;
  TestCopy();
  std::cout << "################### Test Apply ##################" << std::endl;
  TestApply();
  std::cout << "##################### Test To ###################" << std::endl;
  TestToTarget();

  return 0;
}
