/*
 *  cppblas.cpp
 *  cpputils
 *
 *  Created by Alejandro Aragón on 8/4/08.
 *  Copyright 2008 University of Illinois at Urbana-Champaign. All rights reserved.
 *
 */

//#include <cstdlib>
#include "cpp_blas.hpp"

__BEGIN_CPPUTILS__

using std::cerr;

template <>
double algebraic_cast<double>(const dense_vector<double>& v) {
  if (v.size() != 1) {
    cerr<<"*** ERROR *** In algebraic_cast<double>, file "<<__FILE__<<", line "<<__LINE__<<"."<<endl;
    cerr<<"Cannot cast vector of size "<<v.size()<<" into double, aborting..."<<endl;
    exit(1);
  }
  return v(0);
}

template <>
dense_matrix algebraic_cast<dense_matrix>(const dense_vector<double>& v) {
  
  dense_matrix m(v.size(),1);
  for (size_t i=0; i<v.size(); ++i)
    m(i,0) = v(i);
  return m;
}

template <>
double algebraic_cast<double>(const dense_matrix& m) {
  if (m.rows() != 1 || m.columns() != 1) {
    cerr<<"*** ERROR *** In algebraic_cast<double>, file "<<__FILE__<<", line "<<__LINE__<<"."<<endl;
    cerr<<"Cannot cast "<<m.rows()<<"x"<<m.columns()<<" matrix into double, aborting..."<<endl;
    exit(1);
  }
  return m(0,0);
}

template <>
dense_vector<double> algebraic_cast<dense_vector<double> >(const dense_matrix& m) {
  if (m.columns() != 1) {
    cerr<<"*** ERROR *** In algebraic_cast<Vector>, file "<<__FILE__<<", line "<<__LINE__<<"."<<endl;
    cerr<<"Cannot cast "<<m.rows()<<"x"<<m.columns()<<" matrix into vector, aborting..."<<endl;
  }
  dense_vector<double> v(m.columns());
  for (size_t i=0; i<v.size(); ++i)
    v(i) = m(i,0);
  return v;
}

template <>
dense_matrix algebraic_cast<dense_matrix>(double d) {
  return dense_matrix(1,1,d);
}

template <>
dense_vector<double> algebraic_cast<dense_vector<double> >(double d) {
  return dense_vector<double>(1,d);
}


dense_matrix eye(size_t n) {
  
  dense_matrix m(n,n);
  for (size_t i=0; i<n; ++i)
    m(i,i) = 1;
  
  return m;
}

//	template Matrix Gauss2<Matrix>(Matrix&,Matrix&);

//	//			template <class Matrix>
//	Matrix
//	//			typename enable_if<MatrixTraits<Matrix>::conforms, Matrix >::Type
//	Gauss2(Matrix& A, Matrix& b) {
//
//		Matrix copy(A);
//
//		// check if the matrix to factorize is square
//		if (A.rows() != A.columns())
//			throw std::runtime_error("*** ERROR *** Matrix to factorize is not square.");
//
//		// check if the right hand side has compatible dimensions
//		if (A.columns() != b.rows())
//			throw std::runtime_error("*** ERROR *** System with wrong dimensions.");
//
//		// define vector solution
//		Matrix x(b.rows(),b.columns());
//
//		// define variables
//		int tmp = 0;
//		int picked = 0;
//		int loc[A.rows()];
//		double magnitude(0);
//		double tol = 1.0e-15;
//
//		// print matrix A and right hand side vector b
//		cout<<"A ="<<endl;cout<< A<<endl;
//		cout<<"b ="<<endl;cout<< b<<endl;
//
//		// initialize vector loc
//		for(size_t i=0; i<A.rows(); i++)
//			loc[i] = i;
//
//		// carry out gaussian elimination
//		// loop over matrix rows
//		for(size_t i=0; i<A.rows(); ++i){
//			magnitude = 0;
//			// loop over rows to find pivot
//			for(size_t j=i; j<A.rows(); j++){
//				// assing new pivot if magnitude is higher and records pivot location
//				if (std::abs(A(loc[j],i)) > magnitude){
//					magnitude = std::abs(A(loc[j],i));
//					picked = j;
//				}
//			}
//			// checks if a zero pivot was found
//			if(std::abs(magnitude) <= tol){
//				cout<<"maximum pivot found = "<<magnitude<<endl;
//				throw std::runtime_error("*** ERROR *** Singular matrix found.");
//			}
//			tmp = loc[i];
//			loc[i] = loc[picked];
//			loc[picked] = tmp;
//
//			// drive to 0 column i elements in unmarked rows
//			for(size_t j=i+1; j<A.rows(); ++j){
//				//			double t = A(loc[j],i)/A(loc[i],i);
//				A(loc[j],i) = A(loc[j],i)/A(loc[i],i);
//				for(size_t kk = 0; kk<b.columns(); ++kk)
//					b(loc[j],kk) -= b(loc[i],kk)*A(loc[j],i);
//				for(size_t k=i+1; k<A.rows(); ++k){
//					A(loc[j],k) -= A(loc[i],k)*A(loc[j],i);
//				}
//			}
//		}
//
//		// perform back-substitution
//		for(int i=(A.rows()-1); i>=0; i--){
//			for(size_t kk=0; kk<b.columns(); ++kk) {
//				x(i,kk) = b(loc[i],kk)/A(loc[i],i);
//				for(int j=0; j<i; j++){
//					b(loc[j],kk) -= x(i,kk)*A(loc[j],i);
//				}
//			}
//		}
//
//		cout<<"printing b-> "<<b<<endl;
//		cout<<"printing x-> "<<x<<endl;
//
//		cout<<"printing permutation vector"<<endl;
//		for(size_t i=0; i<A.rows(); ++i)
//			cout<<" "<<loc[i];
//		cout<<endl;
//
//		cout<<"printing A"<<A<<endl;
//		Matrix res(A.rows(), A.columns());
//		// carry out multiplication
//		for(size_t i=0; i<A.rows(); ++i)
//			for(size_t j=0; j<A.columns(); ++j)
//				for(size_t k=0; k<=std::min(i,j); ++k) {
//					if(loc[i] == loc[k])
//						res(loc[i],j) += A(loc[k],j);
//					else
//						res(loc[i],j) += A(loc[i],k)*A(loc[k],j);
//				}
//
//		cout<<"res -> "<<res<<endl;
//
//		//	// create LU factors
//		//	Matrix U(A.rows());
//		//	Matrix L(A.rows(),I);
//		//	cout<<"printint L"<<L<<endl;
//		//	cout<<"printing U"<<U<<endl;
//		//
//		//	for(size_t i=0; i<A.rows(); ++i)
//		//		for(size_t j=0; j<A.columns(); ++j)
//		//			if(i>j)
//		//				L(i,j) = A(loc[i],j);
//		//			else
//		//				U(i,j) = A(loc[i],j);
//		//
//		//	cout<<"printint L"<<L<<endl;
//		//	cout<<"printing U"<<U<<endl;
//		//	cout<<"testing L*U -> "<<L*U<<endl;
//
//		return x;
//	}



__END_CPPUTILS__

