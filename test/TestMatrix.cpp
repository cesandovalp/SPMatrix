#include <iostream>
#include <assert.h>

#include <MatrixFactory.hpp>
#include <SerialMatrixOperations.hpp>
#include <MultiCoreMatrixOperations.hpp>

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
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 4, 5 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 4, 5 );

  FillMatrix( A );
  FillMatrix( B );

  // Trace the instructions
  auto C = A + B;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationVector()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 2, 3 );
  std::vector<int> B( { 1, 2, 3 } );

  FillMatrix( A );

  auto C = A * B;

  PrintMatrix( A );
  std::cout << std::endl;
  std::cout << "| 1 |\n| 2 |\n| 3 |" << std::endl << std::endl;
  PrintMatrix( C );
}

void TestMultiplication()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 2, 3 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 4 );

  FillMatrix( A );
  FillMatrix( B );

  auto C = A * B;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationNoResize()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );

  FillMatrix( A );
  FillMatrix( B );

  A.Multiplication( B );

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
}

void TestHadamardProduct()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 10, 10 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 10, 10 );

  FillMatrix( A );
  FillMatrix( B );

  A.HadamardProduct( B );

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
}

void TestMultiplicationIn()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  auto B = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  auto C = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );

  FillMatrix( A );
  FillMatrix( B );

  C.Multiplication( A, B );

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( B );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestMultiplicationScalar()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 2, 3 );

  FillMatrix( A );

  auto B = A;
  auto C = B * 2;

  PrintMatrix( A );
  std::cout << std::endl;
  PrintMatrix( C );
}

void TestTranspose()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 2, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  auto B = !A;
  PrintMatrix( B );
}

void TestMinus()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  auto B = -A;
  PrintMatrix( B );
}

void TestCopy()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

  Matrix<int> B = A;
  PrintMatrix( B );
}

void TestApply()
{
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
  FillMatrix( A );
  PrintMatrix( A );
  std::cout << std::endl;

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
  auto A = sp::MatrixFactory::instance()->CreateMatrix<int>( 3, 3 );
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
  //sp::MatrixFactory::instance()->SetMode( sp::sequential );
  //sp::MatrixFactory::instance()->SetMode( sp::multicore );
  sp::MatrixFactory::instance()->SetMode( sp::gpu );

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
