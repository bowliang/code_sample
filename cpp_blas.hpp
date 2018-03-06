/*
 * C++ BLAS interface using expression templates. Inspired from the work of Todd Veldhuizen
 *
 *  cppblas.hpp
 *  cpputils
 *
 *  Created by Alejandro Aragón on 8/4/08.
 *  Copyright 2008 University of Illinois at Urbana-Champaign. All rights reserved.
 *
 */

#ifndef CPP_BLAS_HPP_
#define CPP_BLAS_HPP_

#include <iostream>

#include "cpputils-config.hpp"

#ifdef HAVE_CBLAS_H
extern "C" {
#include CBLAS_HEADER
}
#endif /* HAVE_CBLAS_H */

#include "util.hpp"
#include "mojo.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "dense_matrix.hpp"
#include "dense_vector.hpp"


//#define CPPUTILS_DEBUG


__BEGIN_CPPUTILS__


using std::cout;
using std::cerr;
using std::endl;

//! Scalar class, used to represent a double literal which appears in the expression.
class Scalar {
private:
  double value_;
public:
  Scalar(double value) : value_(value) {}
  double operator()() const	{
#ifdef CPPUTILS_DEBUG
    cout<<" inside Scalar::operator()"<<endl;
#endif
    return value_; }
  operator double() const {
#ifdef CPPUTILS_DEBUG
    cout<<" inside Scalar::operator double()"<<endl;
#endif
    return value_; }
    friend std::ostream& operator<<(std::ostream& os, const Scalar& s) {
      os<<s.value_; return os;
    }
    };
    
    
    // traits classes
    template <class T, class Enable>
    struct ScalarTraits { static const bool conforms = false; };
    
    template <class T, class Enable>
    struct VectorTraits { static const bool conforms = false; };
    
    template <class T, class Enable>
    struct MatrixTraits { static const bool conforms = false; };
    
    // forward declarations
    class MatrixProxy;
    
    // Gaussian elimination with vector as right hand side
    template <class vector, class matrix>
    typename enable_if<VectorTraits<vector>::conforms && MatrixTraits<matrix>::conforms, fnresult<vector> >::Type
    Gaussian_elimination(const matrix&, const vector&);
    // Gaussian elimination with matrix as right hand side
    template <class matrix>
    typename enable_if<MatrixTraits<matrix>::conforms, fnresult<matrix> >::Type
    Gaussian_elimination(const matrix&, const matrix&);
    // Cholesky factorization with vector as right hand side
    template <class vector, class matrix>
    typename enable_if<VectorTraits<vector>::conforms && MatrixTraits<matrix>::conforms, fnresult<vector> >::Type
    Cholesky_factorization(const matrix&, const vector&);
    // Cholesky factorization with matrix as right hand side
    template <class matrix>
    typename enable_if<MatrixTraits<matrix>::conforms, fnresult<matrix> >::Type
    Cholesky_factorization(const matrix&, const matrix&);
    
    
    template <class A>
    Expr<UnOp<A, TrOp> > inline Transpose(const A&);
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // type definitions to shorten the syntax
    //
    
    // scalar type definitions
    typedef Scalar S;
    typedef dense_vector<> V;
    typedef sparse_vector<> SpV;
    typedef dense_matrix M;
    typedef sparse_matrix<> SpM;
    typedef Expr<UnOp<M, TrOp> > Mtr;
    typedef Expr<UnOp<V, TrOp> > Vtr;
    typedef Expr<UnOp<M, InvOp> > Minv;
    typedef Expr<BinOp<S, V, MulOp> > SVMul;
    typedef Expr<BinOp<S, M, MulOp> > SMMul;
    typedef Expr<BinOp<S, Mtr, MulOp> > SMtrMul;
    typedef Expr<BinOp<Vtr, V, MulOp> > VtrVMul;
    typedef Expr<BinOp<V, Vtr, MulOp> > VVtrMul;
    typedef Expr<BinOp<Minv, V, MulOp> > MinvVMul;
    typedef Expr<BinOp<S, Vtr, MulOp> > SVtrMul;
    typedef Expr<BinOp<V, SVtrMul, MulOp> > VSVtrMul;
    typedef Expr<BinOp<SVMul, Vtr, MulOp> > SVVtrMul;
    typedef Expr<BinOp<Vtr, SVMul, MulOp> > VtrSVMul;
    typedef Expr<BinOp<SVtrMul, V, MulOp> > SVtrVMul;
    typedef Expr<BinOp<SVMul, SVtrMul, MulOp> > SVSVtrMul;
    typedef Expr<BinOp<SVtrMul, SVMul, MulOp> > SVtrSVMul;
    // matrix - vector type definitions
    typedef Expr<BinOp<M, Vtr, MulOp> > MVtrMul;
    typedef Expr<BinOp<M, SVtrMul, MulOp> > MSVtrMul;
    typedef Expr<BinOp<SMMul, Vtr, MulOp> > SMVtrMul;
    typedef Expr<BinOp<SMMul, SVtrMul, MulOp> > SMSVtrMul;
    
    ///////////////////////////////////////////////////////////////////////////////
    // type classes example, useful to implement polymorphism based on concepts
    
    // scalar traits
    
    // partial template specialization for classes that model the scalar concept
    template <>
    struct ScalarTraits<int> {
      static const bool conforms = true;
    };
    template <>
    struct ScalarTraits<double> {
      static const bool conforms = true;
    };
    template <>
    struct ScalarTraits<Scalar> {
      static const bool conforms = true;
    };
    
    // vector traits
    
    template <>
    struct VectorTraits<dense_vector<double> > {
      static const bool conforms = true;
    };
    
    template <>
    struct VectorTraits<sparse_vector<double> > {
      static const bool conforms = true;
    };
    
    
    // matrix traits
    
    template <>
    struct MatrixTraits<dense_matrix> {
      static const bool conforms = true;
    };
    
    template <class T, class Enable = void>
    struct OutputTraits { static const bool conforms = true; };
    
    template <>
    struct OutputTraits<Vtr> {
      static const bool conforms = false;
    };
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Return type metaprogram
    //
    
    template <class A> struct ReturnType {
      typedef A Result;
    };
    template <class,class,class> class BinReturnTraits;
    template <class,class> class UnReturnTraits;
    
    // return types
    template<>
    struct ReturnType<MinvVMul > {
      typedef V Result;
    };
    
    template <class L, class R, class Op>
    struct ReturnType<Expr<BinOp<L,R,Op> > > {
      typedef Expr<BinOp<L,R,Op> > ExpressionType;
      typedef typename ReturnType<L>::Result LeftResult;
      typedef typename ReturnType<R>::Result RightResult;
      typedef typename BinReturnTraits<LeftResult, RightResult, Op >::Result Result;
    };
    
    template <class L, class Op>
    struct ReturnType<Expr<UnOp<L,Op> > > {
      typedef Expr<UnOp<L,Op> > ExpressionType;
      typedef typename ReturnType<L>::Result LeftResult;
      typedef typename UnReturnTraits<LeftResult, Op >::Result Result;
    };
    
    
    //			template <>
    //			struct BinReturnTraits<S,S,AddOp > {
    //				typedef S Result;
    //			};
    
    template <>
    struct BinReturnTraits<double,S,AddOp > {
      typedef S Result;
    };
    
    template <>
    struct BinReturnTraits<S,double,AddOp > {
      typedef S Result;
    };
    
    template <class T>
    struct BinReturnTraits<T,T,AddOp> {
      typedef T Result;
    };
    
    template <class T>
    struct BinReturnTraits<T,T,SubsOp > {
      typedef T Result;
    };
    
    template <>
    struct BinReturnTraits<double,S,SubsOp > {
      typedef S Result;
    };
    template <>
    struct BinReturnTraits<S,double,SubsOp > {
      typedef S Result;
    };
    
    template <>
    struct BinReturnTraits<S,S,MulOp > {
      typedef S Result;
    };
    
    template <>
    struct BinReturnTraits<S,V,MulOp > {
      typedef V Result;
    };
    
    template <>
    struct BinReturnTraits<V,S,MulOp > {
      typedef V Result;
    };
    
    template <>
    struct BinReturnTraits<M,S,MulOp > {
      typedef M Result;
    };
    
    template <>
    struct BinReturnTraits<S,M,MulOp > {
      typedef M Result;
    };
    
    template <>
    struct BinReturnTraits<S,SpV,MulOp > {
      typedef SpV Result;
    };
    
    template <>
    struct BinReturnTraits<M,V,MulOp > {
      typedef V Result;
    };
    
    template <>
    struct BinReturnTraits<V,M,MulOp > {
      typedef M Result;
    };
    
    template <>
    struct BinReturnTraits<M,M,MulOp > {
      typedef M Result;
    };
    
    // return types involving transposed objects
    template <>
    struct UnReturnTraits<M, TrOp > {
      typedef M Result;
    };
    template <>
    struct UnReturnTraits<Mtr, TrOp > {
      typedef M Result;
    };
    template <>
    struct UnReturnTraits<V, TrOp > {
      typedef Vtr Result;
    };
    template <>
    struct UnReturnTraits<Vtr, TrOp > {
      typedef V Result;
    };
    template <>
    struct BinReturnTraits<S,Vtr,MulOp > {
      typedef Vtr Result;
    };
    template <>
    struct BinReturnTraits<V,Vtr,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<Vtr,V,MulOp > {
      typedef double Result;
    };
    template <>
    struct BinReturnTraits<Vtr,M,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<S,Mtr,MulOp > {
      typedef Mtr Result;
    };
    template <>
    struct BinReturnTraits<M,Vtr,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<Mtr,V,MulOp > {
      typedef V Result;
    };
    template <>
    struct BinReturnTraits<V,Mtr,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<Mtr,Vtr,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<M,Mtr,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<Mtr,M,MulOp > {
      typedef M Result;
    };
    template <>
    struct BinReturnTraits<Mtr,Mtr,MulOp > {
      typedef M Result;
    };
    
    // inversion return types
    template <>
    struct UnReturnTraits<M, InvOp > {
      typedef M Result;
    };
    
    template <class A>
    inline std::ostream& Print(std::ostream& os, const Expr<A>& e);
    
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Expression class template
    //
    
    //! Expression class template.
    template<class A>
    class Expr {
      
      A a_;
      
    public:
      typedef A ExpressionType;
      typedef typename ExpressionType::LeftType LeftType;
      typedef typename ExpressionType::RightType RightType;
      typedef typename ExpressionType::OperatorType OperatorType;
      typedef typename ReturnType<Expr<A> >::Result Return;
      
      Expr(const A& x = A())
      : a_(x) {}
      
      LeftType Left() const { return a_.Left(); }
      RightType Right() const { return a_.Right(); }
      
      operator Return() {
#ifdef CPPUTILS_DEBUG
        cout<<"2. Inside Expr::operator Return(). Evaluating expression..."<<endl;
        cout<<"   forwarding evaluation to type:";
        cppfilt(a_);
#endif
        return a_();
      }
      
      Return operator()() const {
#ifdef CPPUTILS_DEBUG
        cout<<"4. Inside Expr::operator()"<<endl;
        cout<<"   type of return type: "<<std::flush;
        cppfilt(Return());
        cout<<"   forwarding function to object of type: "<<std::flush;
        cppfilt(a_);
#endif
        return a_();
      }
      
      double operator()(size_t i) const {
#ifdef CPPUTILS_DEBUG
        cout<<"4. Inside Expr::operator()(size_t)"<<endl;
        cout<<"   forwarding function to object of type: "<<std::flush;
        cppfilt(a_);
#endif
        return a_(i);
      }
      
      double operator()(size_t i, size_t j) const {
#ifdef CPPUTILS_DEBUG
        cout<<"4. Inside Expr::operator()(size_t,size_t)"<<endl;
        cout<<"   forwarding function to object of type: "<<std::flush;
        cppfilt(a_);
#endif
        
        return a_(i,j);
      }
      
      friend inline std::ostream& operator<<(std::ostream& os, const Expr<A>& e)
      { return Print<A>(os,e); }
    };
    
    
    
    /****************************************************************************
     * BinOp represents a binary operation on two expressions.
     * A and B are the two expressions being combined, and Op is an applicative
     * template representing the operation.
     */
    
    template <class T>
    struct ExpTraits {
      typedef const T& Type;
    };
    template<>
    struct ExpTraits<double> {
      typedef double Type;
    };
    template<>
    struct ExpTraits<S> {
      typedef Scalar Type;
    };
    template <class T, class Op>
    struct ExpTraits2 {
      typedef const T& Type;
    };
    template<class Op>
    struct ExpTraits2<double,Op> {
      typedef double Type;
    };
    template<class Op>
    struct ExpTraits2<S,Op> {
      typedef Scalar Type;
    };
    
    
    template<class A, class B, class Op>
    class BinOp {
    public:
      typedef typename ExpTraits2<A,Op>::Type LeftType;
      typedef typename ExpTraits<B>::Type RightType;
      typedef Op OperatorType;
      typedef typename ReturnType< Expr<BinOp> >::Result Return;
    private:
      LeftType a_;
      RightType b_;
    public:
      
      LeftType Left() const { return a_; }
      RightType Right() const { return b_; }
      BinOp(const A& a, const B& b)
      : a_(a), b_(b) {}
      
      BinOp(A& a, const B& b)
      : a_(a), b_(b) {}
      
      inline Return operator()() const {
#ifdef CPPUTILS_DEBUG
        cout<<"3. Inside BinOp"<<endl;
        cout<<"   This type: "<<std::flush;
        cppfilt(*this);
        cout<<"   Left type: "<<std::flush;
        cppfilt(a_);
        cout<<"   Right type: "<<std::flush;
        cppfilt(b_);
        cout<<"   type of return type: "<<std::flush;
        cppfilt(Return());
        cout<<"   forwarding function to Op::apply(a,b) function"<<endl;
#endif
        return Op::apply(a_,b_);
      }
      
      inline double operator()(size_t i, size_t j) const {
#ifdef CPPUTILS_DEBUG
        cout<<"3. Inside BinOp::operator()(i,j)"<<endl;
        cout<<"   This type: "<<std::flush;
        cppfilt(*this);
        cout<<"   Left type: "<<std::flush;
        cppfilt(a_);
        cout<<"   Right type: "<<std::flush;
        cppfilt(b_);
        cout<<"   type of return type: "<<std::flush;
        cppfilt(Return());
        cout<<"   forwarding function to Op::apply(a,b,i,j) function"<<endl;
#endif
        return Op::apply(a_,b_,i,j);
      }
    };
    
    
    template<class A, class Op>
    class UnOp {
      typename ExpTraits2<A,Op>::Type a_;
    public:
      typedef typename ExpTraits2<A,Op>::Type LeftType;
      typedef EmptyType RightType;
      typedef Op OperatorType;
      typedef typename ReturnType< Expr<UnOp> >::Result Return;
      
      typename ExpTraits2<A,Op>::Type Left() const { return a_; }
      UnOp(const A& a)
      : a_(a) {}
      inline Return operator()() const {
#ifdef CPPUTILS_DEBUG
        cout<<"3. Inside UnOp"<<endl;
        cout<<"   This type: "<<std::flush;
        cppfilt(*this);
        cout<<"   Left type: "<<std::flush;
        cppfilt(a_);
        cout<<"   Right type: "<<std::flush;
        cppfilt(RightType());
        cout<<"   type of return type: "<<std::flush;
        cppfilt(Return());
        cout<<"   forwarding function to Op::apply(a) function"<<endl;
#endif
        
        return Op::apply(a_);
      }
      inline double operator()(size_t i, size_t j) const {
#ifdef CPPUTILS_DEBUG
        cout<<"3. Inside UnOp::operator()(i,j)"<<endl;
        cout<<"   This type: "<<std::flush;
        cppfilt(*this);
        cout<<"   Left type: "<<std::flush;
        cppfilt(a_);
        cout<<"   Right type: "<<std::flush;
        cppfilt(RightType());
        cout<<"   type of return type: "<<std::flush;
        cppfilt(Return());
        cout<<"   forwarding function to Op::apply(a,i,j) function"<<endl;
#endif
        
        return Op::apply(a_,i,j);
      }
    };
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // APPLICATIVE TEMPLATE CLASSES
    //
    
    //! Addition operator
    class AddOp {
      
    public:
      AddOp() { }
      
      static inline double apply(double a, double b){
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(double,double)"<<endl;
#endif
        
        return a+b;
      }
      
      static inline V apply(const V& a, const V& b) {
        V r(a);
        for(size_t i=0; i<r.size(); ++i)
          r[i] += b[i];
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(V,V)"<<endl;
#endif
        return r;
      }
      
      static inline SpV apply(const SpV& a, const SpV& b) {
        SpV r(a);
        r += b;
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(SpV,SpV)"<<endl;
#endif
        return r;
      }
      
      static inline M apply(const M& a, const M& b) {
        
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        
        M r(a.rows(),a.columns());
        for(size_t i=0; i<r.rows(); ++i)
          for(size_t j=0; j<r.columns(); ++j)
            r(i,j) = a(i,j) + b(i,j);
        
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M,M)"<<endl;
#endif
        return r;
      }
      
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<A,B,AddOp> > >::Result
      apply(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(any,any)"<<endl;
        cout<<"   Types are not evaluated, addition function is forwarded"<<endl;
#endif
        
        return apply(a,b);
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<Expr<A>,B,AddOp> > >::Result
      apply(const Expr<A>& a, const B& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(Expr<any>,any)"<<endl;
        cout<<"   Evaluating LeftType..."<<endl;
#endif
        
        return apply(a(),b);
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<A,Expr<B>,AddOp> > >::Result
      apply(const A& a, const Expr<B>& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(any,Expr<any>)"<<endl;
        cout<<"   Evaluating RightType..."<<endl;
#endif
        
        return apply(a,b());
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<Expr<A>,Expr<B>,AddOp> > >::Result
      apply(const Expr<A>& a, const Expr<B>& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(Expr<any>,Expr<any>)"<<endl;
        cout<<"   Evaluating LeftType..."<<endl;
        cout<<"   Evaluating RightType..."<<endl;
#endif
        
        return apply(a(),b());
      }
      
      // subscript operators
      
      // one subscript (involving vectors)
      // apply(V,V)
      static inline double apply(const V& x, const V& y, size_t i) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(V,V,i,)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return x(i) + y(i);
      }
      // apply(V,sV)
      static inline double apply(const V& x, const SVMul& B, size_t i) {
        const V& y = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(V,sV,i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return x(i) + B.Left()*y(i);
      }
      // apply(sV,V)
      static inline double apply(const SVMul& A, const V& y, size_t i) {
        const V& x = A.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sV,V,i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return A.Left()*x(i) + y(i);
      }
      // apply(sV,sV)
      static inline double apply(const SVMul& A, const SVMul& B, size_t i) {
        const V& x = A.Right();
        const V& y = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sV,sV,i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return A.Left()*x(i) + B.Left()*y(i);
      }
      // apply(V',V')
      static inline double apply(const Vtr& A, const Vtr& B, size_t i) {
        const V& x = A.Left();
        const V& y = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(V',V',i,)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return x(i) + y(i);
      }
      // apply(V',sV')
      static inline double apply(const Vtr& A, const SVtrMul& B, size_t i) {
        const V& x = A.Left();
        const V& y = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(V',sV',i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return x(i) + B.Left()*y(i);
      }
      // apply(sV',V')
      static inline double apply(const SVtrMul& A, const Vtr& B, size_t i) {
        const V& x = A.Right().Left();
        const V& y = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sV',V',i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return A.Left()*x(i) + y(i);
      }
      // apply(sV',sV')
      static inline double apply(const SVtrMul& A, const SVtrMul& B, size_t i) {
        const V& x = A.Right().Left();
        const V& y = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sV',sV',i)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return A.Left()*x(i) + B.Left()*y(i);
      }
      
      // two subscripts (involving matrices)
      // apply(M,M)
      static inline double apply(const M& a, const M& b, size_t i, size_t j) {
        assert(a.rows() == b.rows() && a.columns() == b.columns());
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M,M,i,j)"<<endl;
#endif
        
        return a(i,j) + b(i,j);
      }
      // apply(M,sM)
      static inline double apply(const M& a, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M,sM,i,j)"<<endl;
#endif
        
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) + B.Left()*b(i,j);
      }
      // apply(M,M')
      static inline double apply(const M& a, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M,M',i,j)"<<endl;
#endif
        
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) + b(j,i);
      }
      // apply(M,sM')
      static inline double apply(const M& a, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M,sM',i,j)"<<endl;
#endif
        
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) + B.Left()*b(j,i);
      }
      // apply(sM,M)
      static inline double apply(const SMMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM,M,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) + b(i,j);
      }
      // apply(sM,sM)
      static inline double apply(const SMMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM,sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) + B.Left()*b(i,j);
      }
      // apply(sM,M')
      static inline double apply(const SMMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM,M',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) + b(j,i);
      }
      // apply(sM,sM')
      static inline double apply(const SMMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM,sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) + B.Left()*b(j,i);
      }
      // apply(M',M)
      static inline double apply(const Mtr& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M',M,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) + b(i,j);
      }
      // apply(M',sM)
      static inline double apply(const Mtr& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) + B.Left()*b(i,j);
      }
      // apply(M',M')
      static inline double apply(const Mtr& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M',M',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) + b(j,i);
      }
      // apply(M',sM')
      static inline double apply(const Mtr& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(M',sM',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) + B.Left()*b(j,i);
      }
      // apply(sM',M)
      static inline double apply(const SMtrMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM',M,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) + b(i,j);
      }
      // apply(sM',sM)
      static inline double apply(const SMtrMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) + B.Left()*b(i,j);
      }
      // apply(sM',M')
      static inline double apply(const SMtrMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM'M',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) + b(j,i);
      }
      // apply(sM',sM')
      static inline double apply(const SMtrMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator+(sM'sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) + B.Left()*b(j,i);
      }
      
    };
    
    
    //! Subtraction operator
    class SubsOp {
      
    public:
      SubsOp() { }
      
      static inline double apply(double a, double b){
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(double,double)"<<endl;
#endif
        
        return a-b;
      }
      
      static inline V apply(const V& a, const V& b) {
        
        assert(a.size() == b.size());
        V r(a);
        for(size_t i=0; i<r.size(); ++i)
          r[i] -= b[i];
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(V,V)"<<endl;
        cout<<"   calling vector copy constructor"<<endl;
        cout<<"5. operator-(V,V) finished evaluation, returning vector"<<endl;
#endif
        return r;
      }
      
      static inline SpV apply(const SpV& a, const SpV& b) {
        SpV r(a);
        r -= b;
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(SpV,SpV)"<<endl;
#endif
        return r;
      }
      
      
      static inline M apply(const M& a, const M& b) {
        
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        
        M r(a.rows(),a.columns());
        for(size_t i=0; i<r.rows(); ++i)
          for(size_t j=0; j<r.columns(); ++j)
            r(i,j) = a(i,j) - b(i,j);
        
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M,M)"<<endl;
        cout<<"   calling matrix parameter constructor"<<endl;
        cout<<"5. operator-(M,M) finished evaluation, returning matrix"<<endl;
#endif
        return r;
      }
      
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<A,B,AddOp> > >::Result
      apply(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(any,any)"<<endl;
        cout<<"   Types are not evaluated, addition function is forwarded"<<endl;
#endif
        
        return apply(a,b);
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<Expr<A>,B,AddOp> > >::Result
      apply(const Expr<A>& a, const B& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(Expr<any>,any)"<<endl;
        cout<<"   Evaluating LeftType..."<<endl;
#endif
        
        return apply(a(),b);
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<A,Expr<B>,AddOp> > >::Result
      apply(const A& a, const Expr<B>& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(any,Expr<any>)"<<endl;
        cout<<"   Evaluating RightType..."<<endl;
#endif
        
        return apply(a,b());
      }
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<Expr<A>,Expr<B>,AddOp> > >::Result
      apply(const Expr<A>& a, const Expr<B>& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(Expr<any>,Expr<any>)"<<endl;
        cout<<"   Evaluating LeftType..."<<endl;
        cout<<"   Evaluating RightType..."<<endl;
#endif
        
        return apply(a(),b());
      }
      
      // subscript operators
      
      // one subscript (involving vectors)
      // apply(V,V)
      static inline double apply(const V& x, const V& y, size_t i) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(V,V,i,)"<<endl;
#endif
        
        assert(x.size() == y.size());
        return x(i) - y(i);
      }
      // apply(V,sV)
      static inline double apply(const V& x, const SVMul& B, size_t i) {
        const V& y = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(V,sV,i)"<<endl;
#endif
        assert(x.size() == y.size());
        return x(i) - B.Left()*y(i);
      }
      // apply(sV,V)
      static inline double apply(const SVMul& A, const V& y, size_t i) {
        const V& x = A.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sV,V,i)"<<endl;
#endif
        assert(x.size() == y.size());
        return A.Left()*x(i) - y(i);
      }
      // apply(sV,sV)
      static inline double apply(const SVMul& A, const SVMul& B, size_t i) {
        const V& x = A.Right();
        const V& y = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sV,sV,i)"<<endl;
#endif
        assert(x.size() == y.size());
        return A.Left()*x(i) - B.Left()*y(i);
      }
      // apply(V',V')
      static inline double apply(const Vtr& A, const Vtr& B, size_t i) {
        const V& x = A.Left();
        const V& y = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(V',V',i,)"<<endl;
#endif
        assert(x.size() == y.size());
        return x(i) - y(i);
      }
      // apply(V',sV')
      static inline double apply(const Vtr& A, const SVtrMul& B, size_t i) {
        const V& x = A.Left();
        const V& y = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(V',sV',i)"<<endl;
#endif
        assert(x.size() == y.size());
        return x(i) - B.Left()*y(i);
      }
      // apply(sV',V')
      static inline double apply(const SVtrMul& A, const Vtr& B, size_t i) {
        const V& x = A.Right().Left();
        const V& y = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sV',V',i)"<<endl;
#endif
        assert(x.size() == y.size());
        return A.Left()*x(i) - y(i);
      }
      // apply(sV',sV')
      static inline double apply(const SVtrMul& A, const SVtrMul& B, size_t i) {
        const V& x = A.Right().Left();
        const V& y = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sV',sV',i)"<<endl;
#endif
        assert(x.size() == y.size());
        return A.Left()*x(i) - B.Left()*y(i);
      }
      
      // two subscripts (involving matrices)
      // apply(M,M)
      static inline double apply(const M& a, const M& b, size_t i, size_t j) {
        assert(a.rows() == b.rows() && a.columns() == b.columns());
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M,M,i,j)"<<endl;
#endif
        return a(i,j) - b(i,j);
      }
      // apply(M,sM)
      static inline double apply(const M& a, const SMMul& B, size_t i, size_t j) {
        
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M,sM,i,j)"<<endl;
#endif
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) - B.Left()*b(i,j);
      }
      // apply(M,M')
      static inline double apply(const M& a, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M,M',i,j)"<<endl;
#endif
        
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) - b(j,i);
      }
      // apply(M,sM')
      static inline double apply(const M& a, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M,sM',i,j)"<<endl;
#endif
        
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(i,j) - B.Left()*b(j,i);
      }
      // apply(sM,M)
      static inline double apply(const SMMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM,M,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) - b(i,j);
      }
      // apply(sM,sM)
      static inline double apply(const SMMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM,sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) - B.Left()*b(i,j);
      }
      // apply(sM,M')
      static inline double apply(const SMMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM,M',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) - b(j,i);
      }
      // apply(sM,sM')
      static inline double apply(const SMMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM,sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(i,j) - B.Left()*b(j,i);
      }
      // apply(M',M)
      static inline double apply(const Mtr& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M',M,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) - b(i,j);
      }
      // apply(M',sM)
      static inline double apply(const Mtr& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) - B.Left()*b(i,j);
      }
      // apply(M',M')
      static inline double apply(const Mtr& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M',M',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) - b(j,i);
      }
      // apply(M',sM')
      static inline double apply(const Mtr& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(M',sM',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return a(j,i) - B.Left()*b(j,i);
      }
      // apply(sM',M)
      static inline double apply(const SMtrMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM',M,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) - b(i,j);
      }
      // apply(sM',sM)
      static inline double apply(const SMtrMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) - B.Left()*b(i,j);
      }
      // apply(sM',M')
      static inline double apply(const SMtrMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM'M',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) - b(j,i);
      }
      // apply(sM',sM')
      static inline double apply(const SMtrMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. inside apply operator-(sM'sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.rows() && a.columns() == b.columns());
        return A.Left()*a(j,i) - B.Left()*b(j,i);
      }
    };
    
    
    
    //! Multiplication operator
    class MulOp {
      
      typedef M MatrixType;
      
    public:
      MulOp() { }
      
      inline double apply(double a, double b) {
        return a*b;
      }
      
      template <class A, class B>
      static inline typename ReturnType<Expr<BinOp<A,B,MulOp> > >::Result
      apply(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"*** WARNING *** MulOp::apply() member function not specialized for classes: "<<endl;
        cout<<"Left type: "<<std::flush;
        cppfilt(a);
        cout<<"Right type: "<<std::flush;
        cppfilt(b);
        cout<<"Trying conversions..."<<endl;
#endif
        return a()*b();
      }
      // apply(s,V)
      template <class A>
      static inline typename enable_if<ScalarTraits<A>::conforms, fnresult<V> >::Type
      apply(const A& s, const V& b) {
        
        V r(b);
        cblas_dscal(b.size(), s, r.data(),1);
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(s,V)"<<endl;
        cout<<"   calling vector copy constructor"<<endl;
        cout<<"4. MulOp::apply(s,V) finished evaluation, returning vector"<<endl;
#endif
        return r;
      }
      
      // apply(s,V)
      template <class A>
      static inline typename enable_if<ScalarTraits<A>::conforms, SpV >::Type
      apply(const A& s, const SpV& b) {
        
        SpV r(b);
        r *= s;
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(s,SpV)"<<endl;
        cout<<"   calling vector copy constructor"<<endl;
        cout<<"4. MulOp::apply(s,SpV) finished evaluation, returning sparse vector"<<endl;
#endif
        return r;
      }
      
      
      // apply(s,M)
      template <class A>
      static inline typename enable_if<ScalarTraits<A>::conforms, fnresult<M> >::Type
      apply(const A& s, const M& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(s,M)"<<endl;
        cout<<"   calling matrix copy constructor"<<endl;
        cout<<"4. MulOp::apply(s,M) finished evaluation, returning matrix"<<endl;
#endif
        
        M r(b);
        cblas_dscal(b.rows()*b.columns(), s, r.data(),1);
        return r;
      }
      
      // apply(s,Mtr)
      template <class A>
      static inline typename enable_if<ScalarTraits<A>::conforms, fnresult<M> >::Type
      apply(const A& s, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(s,Mtr)"<<endl;
        cout<<"   calling matrix constructor"<<endl;
        cout<<"4. MulOp::apply(s,Mtr) finished evaluation, returning matrix"<<endl;
#endif
        
        const M& b = B.Left();
        M r(b.columns(),b.rows());
        for(size_t i=0; i<r.rows(); ++i)
          for(size_t j=0; j<r.columns(); ++j)
            r(i,j) = s*b(j,i);
        return r;
      }
      
      // apply(V,V')
      static inline fnresult<M> apply(const V& x, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,V')"<<endl;
#endif
        // no assertion needed, since the dimensions for this case are alwasy fine
        const V& y = B.Left();
        M r(x.size(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),1.0, x.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(V,sV')
      static inline fnresult<M> apply(const V& x, const SVtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,sV')"<<endl;
#endif
        
        // no assertion needed, since the dimensions for this case are alwasy fine
        const V& y = B.Right().Left();
        M r(x.size(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), B.Left(), x.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,V')
      static inline fnresult<M> apply(const SVMul& A, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,V')"<<endl;
#endif
        
        // no assertion needed, since the dimensions for this case are alwasy fine
        const V& x = A.Right();
        const V& y = B.Left();
        M r(x.size(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left(), x.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,sV')
      static inline fnresult<M> apply(const SVMul& A, const SVtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,sV')"<<endl;
#endif
        
        // no assertion needed, since the dimensions for this case are alwasy fine
        const V& x = A.Right();
        const V& y = B.Right().Left();
        M r(x.size(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left()*B.Left(), x.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(V',V)
      static inline double apply(const Vtr& A, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',V)"<<endl;
#endif
        const V& x = A.Left();
        assert(x.size()==y.size());
        
        return cblas_ddot(y.size(), x.storage(), 1, y.storage(),1);
      }
      // apply(V',sV)
      static inline S apply(const Vtr& A, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',sV)"<<endl;
#endif
        const V& x = A.Left();
        const V& y = B.Right();
        assert(x.size()==y.size());
        
        double r = cblas_ddot(y.size(), x.storage(), 1, y.storage(),1);
        return B.Left()*S(r);
      }
      // apply(sV',V)
      static inline S apply(const SVtrMul& A, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV',V)"<<endl;
#endif
        const V& x = A.Right().Left();
        assert(x.size()==y.size());
        
        double r = cblas_ddot(y.size(), x.storage(), 1, y.storage(),1);
        return A.Left()*S(r);
      }
      // apply(sV',sV)
      static inline S apply(const SVtrMul& A, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV',sV)"<<endl;
#endif
        const V& x = A.Right().Left();
        const V& y = B.Right();
        assert(x.size()==y.size());
        
        double r = cblas_ddot(y.size(), x.storage(), 1, y.storage(),1);
        return A.Left()*B.Left()*S(r);
      }
      
      
      // matrix - vector multiplication
      // apply(M,V)
      static inline fnresult<V> apply(const M& a, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,V)"<<endl;
#endif
        
        assert(a.columns() == y.size());
        V r(a.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, a.rows(), a.columns(),
                    1.0, a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(M,sV)
      static inline fnresult<V> apply(const M& a, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sV)"<<endl;
#endif
        const V& y = B.Right();
        
        assert(a.columns() == y.size());
        V r(a.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, a.rows(), a.columns(),
                    B.Left(), a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(M,V')
      static inline fnresult<M> apply(const M& a, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,V')"<<endl;
#endif
        const V& y = B.Left();
        
        assert(a.columns() == 1);
        M r(a.rows(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),1.0, a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(M,sV')
      static inline fnresult<M> apply(const M& a, const SVtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sV')"<<endl;
#endif
        const V& y = B.Right().Left();
        
        assert(a.columns() == 1);
        M r(a.rows(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),B.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sM,V)
      static inline fnresult<V> apply(const SMMul& A, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,V)"<<endl;
#endif
        const M& a = A.Right();
        
        assert(a.columns() == y.size());
        V r(a.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, a.rows(), a.columns(),
                    A.Left(), a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(sM,sV)
      static inline fnresult<V> apply(const SMMul& A, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sV)"<<endl;
#endif
        const M& a = A.Right();
        const V& y = B.Right();
        
        assert(a.columns() == y.size());
        V r(a.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, a.rows(), a.columns(),
                    A.Left()*B.Left(), a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(sM,V')
      static inline fnresult<M> apply(const SMMul& A, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,V')"<<endl;
#endif
        const M& a = A.Right();
        const V& y = B.Left();
        
        assert(a.columns() == 1);
        M r(a.rows(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sM,sV')
      static inline fnresult<M> apply(const SMMul& A, const SVtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sV')"<<endl;
#endif
        const M& a = A.Right();
        const V& y = B.Right().Left();
        
        assert(a.columns() == 1);
        M r(a.rows(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left()*B.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(M',V)
      static inline fnresult<V> apply(const Mtr& A, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',V)"<<endl;
#endif
        
        const M& a = A.Left();
        assert(a.rows() == y.size());
        V r(a.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, a.rows(), a.columns(),
                    1.0, a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(M',sV)
      static inline fnresult<V> apply(const Mtr& A, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sV)"<<endl;
#endif
        
        const M& a = A.Left();
        const V& y = B.Right();
        assert(a.rows() == y.size());
        V r(a.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, a.rows(), a.columns(),
                    B.Left(), a.storage(), a.rows(), y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(M',V')
      static inline fnresult<M> apply(const Mtr& A, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',V')"<<endl;
#endif
        const M& a = A.Left();
        const V& y = B.Left();
        
        assert(a.rows() == 1);
        M r(a.columns(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), 1.0, a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(M',sV')
      static inline fnresult<M> apply(const Mtr& A, const SVtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sV')"<<endl;
#endif
        const M& a = A.Left();
        const V& y = B.Right().Left();
        
        assert(a.rows() == 1);
        M r(a.columns(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),B.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sM',V)
      static inline fnresult<V> apply(const SMtrMul& A, const V& y) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',V)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        assert(a.rows() == y.size());
        V r(a.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, a.rows(), a.columns(),
                    A.Left(), a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(sM',sV)
      static inline fnresult<V> apply(const SMtrMul& A, const SVMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sV)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const V& y = B.Right();
        assert(a.rows() == y.size());
        V r(a.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, a.rows(), a.columns(),
                    A.Left()*B.Left(), a.storage(), a.rows(),y.storage(), 1, 0, r.data(), 1);
        return r;
      }
      // apply(sM',V')
      static inline fnresult<M> apply(const SMtrMul& A, const Vtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',V')"<<endl;
#endif
        const M& a = A.Right().Left();
        const V& y = B.Left();
        
        assert(a.rows() == 1);
        M r(a.columns(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sM',sV')
      static inline fnresult<M> apply(const SMtrMul& A, const SVtrMul& B) {
        const M& a = A.Right().Left();
        const V& y = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sV')"<<endl;
#endif
        
        assert(a.rows() == 1);
        M r(a.columns(),y.size());
        cblas_dger(CblasColMajor, r.rows(), r.columns(),A.Left()*B.Left(), a.storage(), 1,
                   y.storage(), 1, r.data(), r.rows());
        return r;
      }
      
      // vector - matrix multiplication
      // apply(V,M)
      static inline fnresult<M> apply(const V& y, const M& a) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,M)"<<endl;
#endif
        
        assert(a.rows() == 1);
        M r(y.size(), a.columns());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), 1.0, y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(V,sM)
      static inline fnresult<M> apply(const V& y, const SMMul& B) {
        const M& a = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,sM)"<<endl;
#endif
        
        assert(a.rows() == 1);
        M r(y.size(), a.columns());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), B.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(V,M')
      static inline fnresult<M> apply(const V& y, const Mtr& B) {
        const M& a = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,M')"<<endl;
#endif
        
        assert(a.columns() == 1);
        M r(y.size(), a.rows());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), 1.0, y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(V,sM')
      static inline fnresult<M> apply(const V& y, const SMtrMul& B) {
        const M& a = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V,sM')"<<endl;
#endif
        
        assert(a.columns() == 1);
        M r(y.size(), a.rows());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), B.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,M)
      static inline fnresult<M> apply(const SVMul& A, const M& a) {
        const V& y = A.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,M)"<<endl;
#endif
        
        assert(a.rows() == 1);
        M r(y.size(), a.columns());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), A.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,sM)
      static inline fnresult<M> apply(const SVMul& A, const SMMul& B) {
        const V& y = A.Right();
        const M& a = B.Right();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,sM)"<<endl;
#endif
        
        assert(a.rows() == 1);
        M r(y.size(), a.columns());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), A.Left()*B.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,M')
      static inline fnresult<M> apply(const SVMul& A, const Mtr& B) {
        const V& y = A.Right();
        const M& a = B.Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,M')"<<endl;
#endif
        
        assert(a.columns() == 1);
        M r(y.size(), a.rows());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), A.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      // apply(sV,sM')
      static inline fnresult<M> apply(const SVMul& A, const SMtrMul& B) {
        const V& y = A.Right();
        const M& a = B.Right().Left();
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV,sM')"<<endl;
#endif
        
        assert(a.columns() == 1);
        M r(y.size(), a.rows());
        cblas_dger(CblasColMajor, r.rows(), r.columns(), A.Left()*B.Left(), y.storage(), 1,
                   a.storage(), 1, r.data(), r.rows());
        return r;
      }
      
      // the following functions return matrix objects
      
      // apply(V',M)
      static inline fnresult<M> apply(const Vtr& A, const M& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',M)"<<endl;
#endif
        const V& x = A.Left();
        assert(x.size() == b.rows());
        M r(1, b.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, b.rows(), b.columns(),
                    1.0, b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(V',sM)
      static inline fnresult<M> apply(const Vtr& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',M)"<<endl;
#endif
        const V& x = A.Left();
        const M& b = B.Right();
        assert(x.size() == b.rows());
        M r(1, b.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, b.rows(), b.columns(),
                    B.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      
      // apply(V',M')
      static inline fnresult<M> apply(const Vtr& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',M')"<<endl;
#endif
        const V& x = A.Left();
        const M& b = B.Left();
        
        assert(x.size() == b.columns());
        M r(1,b.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, b.rows(), b.columns(),
                    1.0, b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(V',sM')
      static inline fnresult<M> apply(const Vtr& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',sM')"<<endl;
#endif
        const V& x = A.Left();
        const M& b = B.Right().Left();
        assert(x.size() == b.columns());
        M r(1,b.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, b.rows(), b.columns(),
                    B.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(sV',M)
      static inline fnresult<M> apply(const SVtrMul& A, const M& b) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV',M)"<<endl;
#endif
        const V& x = A.Right().Left();
        assert(x.size() == b.rows());
        M r(1,b.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, b.rows(), b.columns(),
                    A.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(sV',sM)
      static inline fnresult<M> apply(const SVtrMul& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sV',M)"<<endl;
#endif
        const V& x = A.Right().Left();
        const M& b = B.Right();
        assert(x.size() == b.rows());
        M r(1,b.columns());
        cblas_dgemv(CblasColMajor, CblasTrans, b.rows(), b.columns(),
                    A.Left()*B.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(sV',M')
      static inline fnresult<M> apply(const SVtrMul& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',M')"<<endl;
#endif
        const V& x = A.Right().Left();
        const M& b = B.Left();
        
        assert(x.size() == b.columns());
        M r(1,b.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, b.rows(), b.columns(),
                    A.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // apply(sV',sM')
      static inline fnresult<M> apply(const SVtrMul& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(V',sM')"<<endl;
#endif
        const V& x = A.Right().Left();
        const M& b = B.Right().Left();
        assert(x.size() == b.columns());
        M r(1,b.rows());
        cblas_dgemv(CblasColMajor, CblasNoTrans, b.rows(), b.columns(),
                    A.Left()*B.Left(), b.storage(), b.rows(), x.storage(), 1, 0, r.data(), 1);
        return r;
      }
      
      // matrix - matrix multiplication
      
      // apply(M,M)
      static inline fnresult<M> apply(const M& A, const M& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,M)"<<endl;
#endif
        
        assert(A.columns() == B.rows());
        M C(A.rows(),B.columns());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.columns(),
                    A.columns(), 1.0, A.storage(), A.rows(), B.storage(), B.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M,sM)
      static inline fnresult<M> apply(const M& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sM)"<<endl;
#endif
        
        const M& a = A;
        const M& b = B.Right();
        assert(a.columns() == b.rows());
        M C(a.rows(),b.columns());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, a.rows(), b.columns(),
                    a.columns(), B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M,M')
      static inline fnresult<M> apply(const M& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,M')"<<endl;
#endif
        
        const M& b = B.Left();
        assert(A.columns() == b.columns());
        M C(A.rows(),b.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows(), b.rows(),
                    A.columns(), 1.0, A.storage(), A.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M,sM')
      static inline fnresult<M> apply(const M& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sM')"<<endl;
#endif
        
        const M& b = B.Right().Left();
        assert(A.columns() == b.columns());
        M C(A.rows(),b.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows(), b.rows(),
                    A.columns(), B.Left(), A.storage(), A.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM,M)
      static inline fnresult<M> apply(const SMMul& A, const M& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,M)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B;
        assert(a.columns() == b.rows());
        M C(a.rows(),b.columns());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, a.rows(), b.columns(),
                    a.columns(), A.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM,sM)
      static inline fnresult<M> apply(const SMMul& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sM)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right();
        assert(a.columns() == b.rows());
        M C(a.rows(),b.columns());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, a.rows(), b.columns(),
                    a.columns(), A.Left()*B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM,M')
      static inline fnresult<M> apply(const SMMul& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,M')"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Left();
        assert(a.columns() == b.columns());
        M C(a.rows(),b.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, a.rows(), b.rows(),
                    a.columns(), A.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM,sM')
      static inline fnresult<M> apply(const SMMul& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sM')"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right().Left();
        assert(a.columns() == b.columns());
        M C(a.rows(),b.rows());
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, a.rows(), b.rows(),
                    a.columns(), A.Left()*B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M',M)
      static inline fnresult<M> apply(const Mtr& A, const M& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',M)"<<endl;
#endif
        
        const M& a = A.Left();
        assert(a.rows() == B.rows());
        M C(a.columns(),B.columns());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a.columns(), B.columns(),
                    a.rows(), 1.0, a.storage(), a.rows(), B.storage(), B.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M',sM)
      static inline fnresult<M> apply(const Mtr& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sM)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows());
        M C(a.columns(),b.columns());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a.columns(), b.columns(),
                    a.rows(), B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M',M')
      static inline fnresult<M> apply(const Mtr& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',M')"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Left();
        assert(a.rows() == b.columns());
        M C(a.columns(),b.rows());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, a.columns(), b.rows(),
                    a.rows(), 1.0, a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(M',sM')
      static inline fnresult<M> apply(const Mtr& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sM')"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.columns());
        M C(a.columns(),b.rows());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, a.columns(), b.rows(),
                    a.rows(), B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM',M)
      static inline fnresult<M> apply(const SMtrMul& A, const M& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',M)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        assert(a.rows() == B.rows());
        M C(a.columns(),B.columns());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a.columns(), B.columns(),
                    a.rows(), A.Left(), a.storage(), a.rows(), B.storage(), B.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM',sM)
      static inline fnresult<M> apply(const SMtrMul& A, const SMMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sM)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows());
        M C(a.columns(),b.columns());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, a.columns(), b.columns(),
                    a.rows(), A.Left()*B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM',M')
      static inline fnresult<M> apply(const SMtrMul& A, const Mtr& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',M')"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Left();
        assert(a.rows() == b.columns());
        M C(a.columns(),b.rows());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, a.columns(), b.rows(),
                    a.rows(), A.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      // apply(sM',sM')
      static inline fnresult<M> apply(const SMtrMul& A, const SMtrMul& B) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sM')"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.columns());
        M C(a.columns(),b.rows());
        cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, a.columns(), b.rows(),
                    a.rows(), A.Left()*B.Left(), a.storage(), a.rows(), b.storage(), b.rows(),
                    1.0, C.data(), C.rows());
        return C;
      }
      
      
      // operations involving inverse
      template <class T>
      static inline fnresult<V> apply(const Expr<UnOp<T, InvOp> >& A, const V& v) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply Inverse(M)*V"<<endl;
#endif
        
        M a = A.Left()();
        // obtain a copy of right hand side
        V b(v);
        V r = Gaussian_elimination(a,b);
        return r;
      }
      
      // subscript operations
      
      // generic function apply(any,any,i,j) (evaluates both sides before applying the subscripts)
      template <class A, class B>
      static inline double
      apply(const A& a, const B& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"*** WARNING *** MulOp::apply(any, any, i, j) member function not specialized for classes: "<<endl;
        cout<<"Left type: "<<std::flush;
        cppfilt(a);
        cout<<"Right type: "<<std::flush;
        cppfilt(b);
        cout<<"Trying conversions..."<<endl;
#endif
        
        return (a()*b())(i,j);
      }
      
      // apply(s,M,i,j)
      template <class A>
      static inline typename enable_if<ScalarTraits<A>::conforms, double >::Type
      apply(const A& s, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"4. inside MulOp::apply(scalar, M)(i,j)"<<endl;
#endif
        
        return s*b(i,j);
      }
      
      // matrix-matrix multiplication
      // apply(M,M,i,j)
      static inline double apply(const M& a, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,M,i,j)"<<endl;
#endif
        
        assert(a.columns() == b.rows());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(k,j);
        return r;
      }
      // apply(M,sM,i,j)
      static inline double apply(const M& a, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sM,i,j)"<<endl;
#endif
        
        const M& b = B.Right();
        assert(a.columns() == b.rows());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(k,j);
        return B.Left()*r;
      }
      // apply(M,M',i,j)
      static inline double apply(const M& a, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,M',i,j)"<<endl;
#endif
        
        const M& b = B.Left();
        assert(a.columns() == b.columns());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(j,k);
        return r;
      }
      // apply(M,sM',i,j)
      static inline double apply(const M& a, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M,sM',i,j)"<<endl;
#endif
        
        const M& b = B.Right().Left();
        assert(a.columns() == b.columns());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(j,k);
        return B.Left()*r;
      }
      // apply(sM,M,i,j)
      static inline double apply(const SMMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,M,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        assert(a.columns() == b.rows());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(k,j);
        return A.Left()*r;
      }
      // apply(sM,sM,i,j)
      static inline double apply(const SMMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right();
        assert(a.columns() == b.rows());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(k,j);
        return A.Left()*B.Left()*r;
      }
      // apply(sM,M',i,j)
      static inline double apply(const SMMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,M',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Left();
        assert(a.columns() == b.columns());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(j,k);
        return A.Left()*r;
      }
      // apply(sM,sM',i,j)
      static inline double apply(const SMMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM,sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right();
        const M& b = B.Right().Left();
        assert(a.columns() == b.columns());
        double r(0);
        for(size_t k=0; k<a.columns(); ++k)
          r += a(i,k)*b(j,k);
        return A.Left()*B.Left()*r;
      }
      // apply(M',M,i,j)
      static inline double apply(const Mtr& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',M,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        assert(a.rows() == b.rows());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(k,j);
        return r;
      }
      // apply(M',sM,i,j)
      static inline double apply(const Mtr& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(k,j);
        return B.Left()*r;
      }
      // apply(M',M',i,j)
      static inline double apply(const Mtr& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',M',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Left();
        assert(a.rows() == b.columns());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(j,k);
        return r;
      }
      // apply(M',sM',i,j)
      static inline double apply(const Mtr& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(M',sM',i,j)"<<endl;
#endif
        
        const M& a = A.Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.columns());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(j,k);
        return B.Left()*r;
      }
      // apply(sM',M,i,j)
      static inline double apply(const SMtrMul& A, const M& b, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',M,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        assert(a.rows() == b.rows());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(k,j);
        return A.Left()*r;
      }
      // apply(sM',sM,i,j)
      static inline double apply(const SMtrMul& A, const SMMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sM,i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right();
        assert(a.rows() == b.rows());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(k,j);
        return A.Left()*B.Left()*r;
      }
      // apply(sM',M',i,j)
      static inline double apply(const SMtrMul& A, const Mtr& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',M',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Left();
        assert(a.rows() == b.columns());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(j,k);
        return A.Left()*r;
      }
      // apply(sM',sM',i,j)
      static inline double apply(const SMtrMul& A, const SMtrMul& B, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"5. Inside apply(sM',sM',i,j)"<<endl;
#endif
        
        const M& a = A.Right().Left();
        const M& b = B.Right().Left();
        assert(a.rows() == b.columns());
        double r(0);
        for(size_t k=0; k<a.rows(); ++k)
          r += a(k,i)*b(j,k);
        return A.Left()*B.Left()*r;
      }
    };
    
    template<>
    inline S MulOp::apply<S,S>(const S& a, const S& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"4. inside specialized MulOp::apply<S,S>()"<<endl;
#endif
      
      return S(a*b);
    }
    
    // DApDivide -- divide two doubles
    class DApDivide {
    public:
      DApDivide() { }
      
      static inline double apply(double a, double b)
      { return a/b; }
    };
    
    // TrOp -- transpose
    struct TrOp {
      
      static inline const V& apply(const V& v) { return v; }
      static inline M apply(const M& A) {
#ifdef CPPUTILS_DEBUG
        cout<<"inside TrOp::apply(M)"<<endl;
#endif
        
        M r(A.columns(),A.rows());
        for(size_t i=0; i<r.rows(); ++i)
          for(size_t j=0; j<r.columns(); ++j)
            r(i,j) = A(j,i);
        return r;
      }
      static inline double apply(const M& A, size_t i, size_t j) {
#ifdef CPPUTILS_DEBUG
        cout<<"inside TrOp::apply(M,i,j)"<<endl;
#endif
        
        return A(j,i);
      }
    };
    
    // InvOp, Inversion operator
    struct InvOp {
      
      static inline M apply(const M& m) {
        // create unit matrix
        M unit(m.rows(), M::I);
        // call Gaussian elimination
        M r = Gaussian_elimination(m,unit);
        return r;
      }
    };
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // OPERATORS
    //
    
    // Unary plus operator
    
    //! Operator+(any)
    template<class A>
    typename enable_if<!ScalarTraits<A>::conforms, A&>::Type
    inline operator+(A& a) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside unary operator+(any)"<<endl;
      cout<<"   Type: "<<std::flush;
      cppfilt(a);
#endif
      return a;
    }
    
    // Addition operators
    
    //! Operator+(any,any), no scalars
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<A, B, AddOp> > >::Type
    inline operator+(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator+(any,any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<A, B, AddOp> ExprT;
      return Expr<ExprT>(ExprT(a,b));
    }
    
    //! Operator+(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, double>::Type
    inline operator+(A a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator+(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      return static_cast<double>(a)+static_cast<double>(b);
    }
    
    //! Operator+(any, scalar)
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, Expr<BinOp<S, A, AddOp> > >::Type
    inline operator+(const A& a, B s) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator+(any, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(s);
#endif
      
      typedef BinOp<S, A, AddOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),a));
    }
    
    //! Operator+(scalar, any)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<S, B, AddOp> > >::Type
    inline operator+(A s, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator+(scalar, any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(s);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<S, B, AddOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),b));
    }
    
    //! Operator+=(any,any), no scalars
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, A&> ::Type
    inline operator+=(A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator+=(any,any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      a += b();
      return a;
    }
    
    //! Operator+=(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, A&>::Type
    inline operator+=(A& a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator+(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      a += static_cast<double>(b);
      return a;
    }
    
    //! Operator+=(matrix,vector), no scalars
    inline void operator+=(dense_matrix& a, const V& b) {
      // call static check to give an error at compilation time
      assert(false);
    }
    
    //! Operator-=(vector,matrix), no scalars
    inline void operator+=(V& a, const dense_matrix& b) {
      assert(false);
    }
    
    // Unary minus operator
    
    //! Operator-(any)
    template<class A>
    typename enable_if<!ScalarTraits<A>::conforms, Expr<BinOp<S, A, MulOp> > >::Type
    inline operator-(const A& a) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside unary operator-(any)"<<endl;
      cout<<"   Type: "<<std::flush;
      cppfilt(a);
#endif
      
      typedef BinOp<S, A, MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(-1.0),a));
    }
    
    // Substraction operators
    
    //! Operator-(any,any), no scalars
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<A, B, SubsOp> > >::Type
    inline operator-(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator-(any,any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<A, B, SubsOp> ExprT;
      return Expr<ExprT>(ExprT(a,b));
    }
    
    //! Operator-(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, double>::Type
    inline operator-(A a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator-(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      return static_cast<double>(a)-static_cast<double>(b);
    }
    
    //! Operator-(any, scalar)
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, Expr<BinOp<S, A, SubsOp> > >::Type
    inline operator-(const A& a, B s) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator-(any, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(s);
#endif
      
      typedef BinOp<S, A, SubsOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),a));
    }
    
    //! Operator-(scalar, any)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<S, B, SubsOp> > >::Type
    inline operator-(A s, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator-(scalar, any)"<<endl;
      cout<<"   Left:";
      cppfilt(s);
      cout<<"   Right:";
      cppfilt(b);
#endif
      
      typedef BinOp<S, B, SubsOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),b));
    }
    
    //! Operator-=(any,any), no scalars
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, A&> ::Type
    inline operator-=(A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator-=(any,any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      a -= b();
      return a;
    }
    
    //! Operator-=(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, A&>::Type
    inline operator-=(A& a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator-=(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      a -= static_cast<double>(b);
      return a;
    }
    
    
    // Multiplication operators
    
    //! Operator*(any,any), no scalars
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<A, B, MulOp> > >::Type
    inline operator*(const A& a, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator*(any,any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<A, B, MulOp> ExprT;
      return Expr<ExprT>(ExprT(a,b));
    }
    
    //! Operator*(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, double>::Type
    inline operator*(A a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator*(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      return static_cast<double>(a)*static_cast<double>(b);
    }
    
    //! Operator*(any, scalar)
    template<class A,class B>
    typename enable_if<!ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, Expr<BinOp<S, A, MulOp> > >::Type
    inline operator*(const A& a, B s) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator*(any, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(s);
#endif
      
      typedef BinOp<S, A, MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),a));
    }
    
    //! Operator*(scalar, any)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<S, B, MulOp> > >::Type
    inline operator*(A s, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator*(scalar, any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(s);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<S, B, MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(static_cast<double>(s)),b));
    }
    
    // partial specialization for when one of the arguments is a literal
    // (change the value of the literal)
    //! Operator*(double, Expr(Scalar,any))
    template <class A>
    Expr<BinOp<Scalar,A,MulOp> >
    inline operator*(double s, const Expr<BinOp<Scalar,A,MulOp> >& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator*(double,double*any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(s);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<Scalar,A,MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(s*b.Left()),b.Right()));
    }
    // partial specialization for when one of the arguments is a literal
    // (change the value of the literal)
    //! Operator*(Expr(Scalar,any), double)
    template <class A>
    Expr<BinOp<Scalar,A,MulOp> >
    inline operator*(const Expr<BinOp<Scalar,A,MulOp> >& a, double s) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator*(double*any,double)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(s);
#endif
      
      typedef BinOp<Scalar,A,MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(s*a.Left()),a.Right()));
    }
    
    
    
    // Division operators
    
    //! Operator*(scalar, scalar)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && ScalarTraits<B>::conforms, double>::Type
    inline operator/(A a, B b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside general operator/(scalar, scalar)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(a);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      return static_cast<double>(a)/static_cast<double>(b);
    }
    
    
    //! Operator/(scalar, any)
    template<class A,class B>
    typename enable_if<ScalarTraits<A>::conforms && !ScalarTraits<B>::conforms, Expr<BinOp<S, B, MulOp> > >::Type
    inline operator/(A s, const B& b) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside operator/(scalar, any)"<<endl;
      cout<<"   Left: "<<std::flush;
      cppfilt(s);
      cout<<"   Right: "<<std::flush;
      cppfilt(b);
#endif
      
      typedef BinOp<S, B, MulOp> ExprT;
      return Expr<ExprT>(ExprT(S(1.0/static_cast<double>(s)),b));
    }
    
    // Transposition operators
    template <class A>
    Expr<UnOp<A, TrOp> >
    inline Transpose(const A& a) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside Transpose(any)"<<endl;
      cout<<"   Type: "<<std::flush;
      cppfilt(a);
#endif
      
      typedef UnOp<A, TrOp> ExprT;
      return Expr<ExprT>(ExprT(a));
    }
    template <class A>
    inline const A& Transpose(const Expr<UnOp<A, TrOp> >& a) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside Transpose(Transpose(any))"<<endl;
      cout<<"   Type: "<<std::flush;
      cppfilt(a);
#endif
      
      return a.Left();
    }
    // partial template specialization for scalar
    inline S Transpose(const S& s) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside Transpose(Scalar)"<<endl;
      cout<<"   type: "<<std::flush;
      cppfilt(s);
#endif
      
      return s;
    }
    
    // inversion operators
    template <class A>
    Expr<UnOp<A,InvOp> >
    inline Inverse(const A& a) {
#ifdef CPPUTILS_DEBUG
      cout<<"1. Inside Inverse(any)"<<endl;
      cout<<"   Type: "<<std::flush;
      cppfilt(a);
#endif
      typedef UnOp<A, InvOp> ExprT;
      return Expr<ExprT>(ExprT(a));
    }
    
    template <class A>
    inline std::ostream& Print(std::ostream& os, const Expr<A>& e) {
      os<<e();
      return os;
    }
    
    template<>
    inline std::ostream& Print<UnOp<V, TrOp> >(std::ostream& os, const Vtr& B) {
      const V& y = B.Left();
      for(size_t i=0; i<y.size(); ++i)
        os<<" "<<y[i];
      os<<endl;
      return os;
    }
    
    
    //! \relates Matrix Kronecker product.
    /*! This function creates a block matrix result of the Kronecker product.
     * \param A - A constant reference to the first matrix of the operation.
     * \param B - A constant reference to the second matrix of the operation.
     * \return A matrix, result of the Kronecker product between the two matrices.
     */
    template <class Matrix>
    typename enable_if<MatrixTraits<Matrix>::conforms, Matrix >::Type
    Kron(const Matrix& A, const Matrix& B) {
      
      Matrix r(A.rows()*B.rows(),A.columns()*B.columns());
      for (size_t i=0; i<A.rows(); ++i) {
        for (size_t j=0; j<A.columns(); ++j) {
          for (size_t k=0; k<B.rows(); ++k) {
            for (size_t l=0; l<B.columns(); ++l) {
              r(i*B.rows()+k, j*B.columns()+l) += A(i,j)*B(k,l);
            }
          }
        }
      }
      return r;
    }
    
    template <class Matrix, class Vector>
    typename enable_if<MatrixTraits<Matrix>::conforms && VectorTraits<Vector>::conforms, Matrix >::Type
    Kron(const Vector& A, const Matrix& B) {
      
      Matrix r(A.size()*B.rows(),B.columns());
      for (size_t i=0; i<A.size(); ++i) {
        for (size_t k=0; k<B.rows(); ++k) {
          for (size_t l=0; l<B.columns(); ++l) {
            r(i*B.rows()+k, l) += A(i)*B(k,l);
          }
        }
      }
      return r;
    }
    
    template <class Matrix, class Vector>
    typename enable_if<MatrixTraits<Matrix>::conforms && VectorTraits<Vector>::conforms, Matrix >::Type
    Kron(const Matrix& A, const Vector& B) {
      
      Matrix r(A.rows()*B.size(),A.columns());
      for (size_t i=0; i<A.rows(); ++i) {
        for (size_t j=0; j<A.columns(); ++j) {
          for (size_t k=0; k<B.size(); ++k) {
            r(i*B.size()+k, j) += A(i,j)*B(k);
          }
        }
      }
      return r;
    }
    
    template <class Vector>
    typename enable_if<VectorTraits<Vector>::conforms, Vector >::Type
    Kron(const Vector& a, const Vector& b) {
      
      Vector r(a.size()*b.size());
      for (size_t i=0; i<a.size(); ++i) {
        for (size_t k=0; k<b.size(); ++k) {
          r(i*b.size()+k) += a(i)*b(k);
        }
      }
      return r;
    }
    
    // special cases where vectors are transposed
    
    template <class Matrix>
    typename enable_if<MatrixTraits<Matrix>::conforms, Matrix >::Type
    Kron(const Vtr& a, const Matrix& B) {
      
      const V& A = a.Left();
      Matrix r(B.rows(),A.size()*B.columns());
      for (size_t j=0; j<A.size(); ++j) {
        for (size_t k=0; k<B.rows(); ++k) {
          for (size_t l=0; l<B.columns(); ++l) {
            r(k, j*B.columns()+l) += A(j)*B(k,l);
          }
        }
      }
      return r;
    }
    
    template <class Matrix>
    typename enable_if<MatrixTraits<Matrix>::conforms, Matrix >::Type
    Kron(const Matrix& A, const Vtr& b) {
      
      const V& B = b.Left();
      
      Matrix r(A.rows(),A.columns()*B.size());
      for (size_t i=0; i<A.rows(); ++i) {
        for (size_t j=0; j<A.columns(); ++j) {
          for (size_t l=0; l<B.size(); ++l) {
            r(i, j*B.size()+l) += A(i,j)*B(l);
          }
        }
      }
      return r;
    }
    
    
    ////////////////////////////////////////////////////////////////////////
    /// Solution to system of linear equations
    
    //! Class used to store the result of the LU factorization of a matrix.
    /*! The class contains two member variables, a dense matrix that is used to
     * store the result of the LU factorization of a matrix, and the vector
     * containing the pivoting information. The two matrices can be stored
     * in a single matrix because the diagonal elements of the lower factor
     * are equal to 1. The class has moving constructors to avoid the creation
     * of temporaries. The class also implements a solve() function, that is
     * used to obtain the solution to a system of linear equations for which the
     * factors L and U were obtained using forward and backward substitution.
     */
    struct LU_storage : public enabled<LU_storage> {
      
      typedef dense_vector<int> vector_type;
      
      dense_matrix lu_;
      vector_type loc_;
      
      bool Singular_;
      
      LU_storage(const dense_matrix& A) : lu_(A), loc_(A.rows()), Singular_(false) {
        
      }
      
      LU_storage(fnresult<LU_storage> src) { // source is a fnresult
        // move lu data
        lu_.data_ = src.lu_.data_;
        src.lu_.data_ = 0;
        lu_.m_ = src.lu_.m_;
        lu_.n_ = src.lu_.n_;
        // move loc data
        loc_.data_ = src.loc_.data_;
        src.loc_.data_ = 0;
        loc_.n_ = src.loc_.n_;
      }
      
      LU_storage(temporary<LU_storage> src) { // source is a temporary
        LU_storage& rhs = src.get();
        // move lu data
        lu_.data_ = rhs.lu_.data_;
        rhs.lu_.data_ = 0;
        lu_.m_ = rhs.lu_.m_;
        lu_.n_ = rhs.lu_.n_;
        // move loc data
        loc_.data_ = rhs.loc_.data_;
        rhs.loc_.data_ = 0;
        loc_.n_ = rhs.loc_.n_;
      }
      
      template <class vector>
      fnresult<vector> solve(const vector& b) {
        
        vector c(b);
        vector x(c.size());
        
        // perform forward substitution
        // loop over columns
        for(size_t j=0; j<lu_.columns(); ++j) {
          // compute solution component
          // note: the computation below is actually 
          // x[j] = c[j]/lu_(loc_[j],j) but the denominator in this case is 1.0
          x[j] = c[j];
          // loop over rows below the diagonal
          for(size_t i=j+1; i<lu_.columns(); ++i)
            // update right hand side
            c[i] -= lu_(loc_[i],j)*x[j];
        }
        
        // perform back-substitution and store solution in right hand side vector
        // loop backwards over columns
        for(int j=(lu_.rows()-1); j>=0; --j){
          // compute solution component
          c[j] = x[j]/lu_(loc_[j],j);
          // loop over rows above the diagonal
          for(int i=0; i<j; ++i)
            // update right hand side
            x[i] -= c[j]*lu_(loc_[i],j);
        }
        return c;
      }
    };
    
    //! Factorize a matrix into its lower and triangular factors
    /*! This function carries out LU factorization of the matrix passed
     * as a parameter to the function. The factores are stored in a
     * single matrix inside the LU_storage structure, together with
     * the vector storing the pivoting information. The LU_storage
     * class can be subsequently used to solve systems of linear
     * equations through forward and backward substitution.
     */
    template <class matrix_type>
    fnresult<LU_storage> LU(const matrix_type& a, bool& Singular) {
      
      LU_storage lu(a);
      
      dense_matrix& A = lu.lu_;
      LU_storage::vector_type& loc = lu.loc_;
      
      bool& IsSingular = lu.Singular_;
      
      // check if the matrix to factorize is square
      assert(A.rows() == A.columns());
      
      // define variables
      int tmp = 0;
      int picked = 0;
      double magnitude(0);
      double tol = 1.0e-15;
      
      // initialize vector loc
      for(size_t i=0; i<A.rows(); i++)
        loc[i] = i;
      
      // carry out gaussian elimination
      // loop over matrix rows
      for(size_t i=0; i<A.rows(); ++i){
        magnitude = 0;
        // loop over rows to find pivot
        for(size_t j=i; j<A.rows(); j++){
          // assing new pivot if magnitude is higher and records pivot location
          if (std::abs(A(loc[j],i)) > magnitude){
            magnitude = std::abs(A(loc[j],i));
            picked = j;
          }
        }
        // checks if a zero pivot was found
        if(std::abs(magnitude) <= tol){
          //cout<<"maximum pivot found = "<<magnitude<<endl;
          Singular  = true;
          IsSingular = true ;       
          //throw std::runtime_error("*** ERROR *** Singular matrix found.");
        }
        tmp = loc[i];
        loc[i] = loc[picked];
        loc[picked] = tmp;
        
        // drive to 0 column i elements in unmarked rows
        for(size_t j=i+1; j<A.rows(); ++j){
          A(loc[j],i) = A(loc[j],i)/A(loc[i],i);
          for(size_t k=i+1; k<A.rows(); ++k){
            A(loc[j],k) -= A(loc[i],k)*A(loc[j],i);
          }
        }
      }
      
      
      
      return lu;
    }
    
    //! Factorize a matrix into its lower and triangular factors
    /*! This function carries out LU factorization of the matrix passed
     * as a parameter to the function. The factores are stored in a
     * single matrix inside the LU_storage structure, together with
     * the vector storing the pivoting information. The LU_storage
     * class can be subsequently used to solve systems of linear
     * equations through forward and backward substitution.
     */
    template <class matrix_type>
    fnresult<LU_storage> LU(const matrix_type& a) {
      
      LU_storage lu(a);
      dense_matrix& A = lu.lu_;
      LU_storage::vector_type& loc = lu.loc_;
      
      // check if the matrix to factorize is square
      assert(A.rows() == A.columns());
      
      // define variables
      int tmp = 0;
      int picked = 0;
      double magnitude(0);
      double tol = 1.0e-15;
      
      // initialize vector loc
      for(size_t i=0; i<A.rows(); i++)
        loc[i] = i;
      
      // carry out gaussian elimination
      // loop over matrix rows
      for(size_t i=0; i<A.rows(); ++i){
        magnitude = 0;
        // loop over rows to find pivot
        for(size_t j=i; j<A.rows(); j++){
          // assing new pivot if magnitude is higher and records pivot location
          if (std::abs(A(loc[j],i)) > magnitude){
            magnitude = std::abs(A(loc[j],i));
            picked = j;
          }
        }
        // checks if a zero pivot was found
        if(std::abs(magnitude) <= tol){
          cout<<"maximum pivot found = "<<magnitude<<endl;
          throw std::runtime_error("*** ERROR *** Singular matrix found.");
        }
        tmp = loc[i];
        loc[i] = loc[picked];
        loc[picked] = tmp;
        
        // drive to 0 column i elements in unmarked rows
        for(size_t j=i+1; j<A.rows(); ++j){
          A(loc[j],i) = A(loc[j],i)/A(loc[i],i);
          for(size_t k=i+1; k<A.rows(); ++k){
            A(loc[j],k) -= A(loc[i],k)*A(loc[j],i);
          }
        }
      }
      return lu;
    }
    
    
    //! \relates Matrix
    /*! \brief Linear system of equation solver by Gaussian elimination
     *
     * Solves a system of linear equations using Gaussian elimination with
     * partial pivoting.
     * \param A - Matrix object
     * \param b - Vector object
     * \return A Vector object solution.
     */
    template <class vector, class matrix>
    typename enable_if<VectorTraits<vector>::conforms && MatrixTraits<matrix>::conforms, fnresult<vector> >::Type
    Gaussian_elimination(const matrix& a, const vector& B) {
      
      // check if the matrix to factorize is square
      assert(a.rows() == a.columns());
      
      // check if the right hand side has compatible dimensions
      assert(a.columns() == B.size());
      
      // define vector solution
      matrix A(a);
      vector b(B);
      vector x(b.size());
      
      // define variables
      int tmp = 0;
      int picked = 0;
      double magnitude(0);
      double tol = 1.0e-15;
      
      // initialize vector loc
      dense_vector<int> loc(A.rows());
      for(size_t i=0; i<A.rows(); i++)
        loc[i] = i;
      
      // carry out gaussian elimination
      double t;
      // loop over matrix rows
      for(size_t i=0; i<A.rows(); i++){
        magnitude = 0;
        // loop over rows to find pivot
        for(size_t j=i; j<A.rows(); j++){
          // assing new pivot if magnitude is higher and records pivot location
          if (std::abs(A(loc[j],i)) > magnitude){
            magnitude = std::abs(A(loc[j],i));
            picked = j;
          }
        }
        // checks if a zero pivot was found
        if(std::abs(magnitude) <= tol){
          cout<<"maximum pivot found = "<<magnitude<<endl;
          throw std::runtime_error("*** ERROR *** Singular matrix found.");
        }
        tmp = loc[i];
        loc[i] = loc[picked];
        loc[picked] = tmp;
        
        // drive to 0 column i elements in unmarked rows
        for(size_t j=i+1; j<A.rows(); j++){
          t = A(loc[j],i)/A(loc[i],i);
          b[loc[j]] = b[loc[j]] - b[loc[i]]*t;
          for(size_t k=i+1; k<A.rows(); k++){
            A(loc[j],k) -= A(loc[i],k)*t;
          }
        }
      }
      
      // perform back-substitution
      for(int i=(A.rows()-1); i>=0; i--){
        x[i] = b[loc[i]]/A(loc[i],i);
        for(int j=0; j<i; j++)
          b[loc[j]] -= x[i]*A(loc[j],i);
      }
      return x;
    }
    
    //! \relates Matrix
    /*! \brief Linear system of equation solver by Gaussian elimination
     *
     * Solves a system of linear equations using Gaussian elimination
     * with partial pivoting.
     * \param A - Matrix object
     * \param b - Matrix object, containing n right hand sides
     * \return A Matrix object containing vector solutions.
     */
    template <class matrix>
    typename enable_if<MatrixTraits<matrix>::conforms, fnresult<matrix> >::Type
    Gaussian_elimination(const matrix& a, const matrix& B) {
      
      // check if the matrix to factorize is square
      assert(a.rows() == a.columns());
      
      // check if the right hand side has compatible dimensions
      assert(a.columns() == B.rows());
      
      // define vector solution
      matrix A(a);
      matrix b(B);
      matrix x(b.rows(),b.columns());
      
      // define variables
      int tmp = 0;
      int picked = 0;
      double magnitude = 0;
      double tol = 1.0e-15;
      
      // initialize vector loc
      dense_vector<int> loc(A.rows());
      for(size_t i=0; i<A.rows(); i++)
        loc[i] = i;
      
      // carry out gaussian elimination
      // loop over matrix rows
      for(size_t i=0; i<A.rows(); ++i){
        magnitude = 0;
        // loop over rows to find pivot
        for(size_t j=i; j<A.rows(); j++){
          // assing new pivot if magnitude is higher and records pivot location
          if (std::abs(A(loc[j],i)) > magnitude){
            magnitude = std::abs(A(loc[j],i));
            picked = j;
          }
        }
        // checks if a zero pivot was found
        if(std::abs(magnitude) <= tol){
          cout<<"maximum pivot found = "<<magnitude<<endl;
          throw std::runtime_error("*** ERROR *** Singular matrix found.");
        }
        tmp = loc[i];
        loc[i] = loc[picked];
        loc[picked] = tmp;
        
        // drive to 0 column i elements in unmarked rows
        for(size_t j=i+1; j<A.rows(); ++j){
          A(loc[j],i) = A(loc[j],i)/A(loc[i],i);
          for(size_t kk = 0; kk<b.columns(); ++kk)
            b(loc[j],kk) -= b(loc[i],kk)*A(loc[j],i);
          for(size_t k=i+1; k<A.rows(); ++k){
            A(loc[j],k) -= A(loc[i],k)*A(loc[j],i);
          }
        }
      }
      
      // perform back-substitution
      for(int i=(A.rows()-1); i>=0; i--){
        for(size_t kk=0; kk<b.columns(); ++kk) {
          x(i,kk) = b(loc[i],kk)/A(loc[i],i);
          for(int j=0; j<i; j++){
            b(loc[j],kk) -= x(i,kk)*A(loc[j],i);
          }
        }
      }
      return x;
    }
    
    //! \relates Matrix
    /*! \brief Linear system of equation solver by Cholesky decomposition
     * Solves a system of linear equations using Cholesky decomposition.
     * \param A - Matrix object
     * \param b - Vector object
     * \return A Vector object solution.
     */
    template <class vector, class matrix>
    typename enable_if<VectorTraits<vector>::conforms && MatrixTraits<matrix>::conforms, fnresult<vector> >::Type
    Cholesky_factorization(const matrix& a, const vector& B) {
      
      // check if the matrix to factorize is square
      assert(a.rows() == a.columns());
      
      // check if the right hand side has compatible dimensions
      assert(a.columns() == B.size());
      
      // check that the matrix is indeed symmetric
      //				assert(a.is_symmetric());
      
      // check for matrix symmetry
      for (size_t i=1; i<a.rows(); ++i)
        for (size_t j=0; j<i; ++j)
          if ( std::abs(a(i,j) - a(j,i)) > 1.0e-10) {
            cout<<"*** ERROR *** Cholesky factorization can only be performed on symmetric matrices."<<endl;
            cout<<"*** ABORTING ***"<<endl;
            exit(1);
          }
      
      
      
      // define matrices and vectors
      matrix A(a);
      vector b(B);
      vector x(b.size());
      vector y(b.size());
      
      // carry out Cholesky factorization
      //    the result of the Cholesky factorization is a lower triangular matrix
      //    that is stored in the lower triangular portion of the original matrix.
      //    even though the upper triangular elements (excluding diagonal elements)
      //    remain in the original matrix, they are not used in further
      //    computations.  thus, the lower triangular portion of the resulting
      //    matrix is such that A = L*L' where L' denotes the transpose of the
      //    lower triangular matrix L
      // loop over columns of the matrix
      for(size_t k=0; k<A.columns(); ++k){
        A(k,k) = std::sqrt(A(k,k));
        // scale current column
        for(size_t i=k+1; i<A.columns(); ++i)
          A(i,k) = A(i,k)/A(k,k);
        // from each remaining column, subtract multiple of current column
        for(size_t j=k+1; j<A.columns(); ++j)
          for(size_t i=j; i<A.columns(); ++i)
            A(i,j) -= A(i,k)*A(j,k);
      }
      
      // perform forward substitution on the system L*y = b, where y = L'*x and
      //    L' denotes the transpose of the lower triangular matrix L stored in A
      // loop over columns
      for(size_t j=0; j<A.columns(); ++j){
        
        // stop if matrix is singular
        if(A(j,j) == 0)
          throw std::runtime_error("*** ERROR *** Singular matrix found.");
        // compute solution component
        y(j) = b(j)/A(j,j);
        // update right hand side vector
        for(size_t i=j+1; i<A.columns(); ++i)
          b(i) -= y(j)*A(i,j);
      }
      
      // perform back substitution on the system L'*x = y to obtain the final
      //    soluion vector x
      // loop backwards over columns
      for(int j=(A.columns()-1); j>=0; --j){
        // compute solution component
        x(j) = y(j)/A(j,j);
        // update right hand side vector
        for(int i=0; i<j; ++i)
          y(i) -= x(j)*A(j,i);
      }
      
      // note that the code for back substitution access the lower triangular part
      //    of the matrix
      
      return x;
    }
    
    //! \relates Matrix
    /*! \brief Linear system of equation solver by Cholesky decomposition
     *     Solves a system of linear equations using Cholesky decomposition.
     *     \param A - Matrix object
     *     \param b - Vector object
     *     \return A Vector object solution.
     */
    template <class matrix>
    typename enable_if<MatrixTraits<matrix>::conforms, fnresult<matrix> >::Type
    Cholesky_factorization(const matrix& a, const matrix& B) {
      
      // check if the matrix to factorize is square
      assert(a.rows() == a.columns());
      
      // check if the right hand side has compatible dimensions
      assert(a.columns() == B.rows());
      
      // check that the matrix is indeed symmetric
      //				assert(a.is_symmetric());
      
      // check for matrix symmetry
      for (size_t i=1; i<a.rows(); ++i)
        for (size_t j=0; j<i; ++j)
          if ( std::abs(a(i,j) - a(j,i)) > 1.0e-10) {
            cout<<"*** ERROR *** Cholesky factorization can only be performed on symmetric matrices."<<endl;
            cout<<"*** ABORTING ***"<<endl;
            exit(1);
          }
      
      // define matrices
      matrix A(a);
      matrix b(B);
      matrix x(b.rows(),b.columns());
      matrix y(b.rows(),b.columns());
      
      // carry out Cholesky factorization
      //    the result of the Cholesky factorization is a lower triangular matrix
      //    that is stored in the lower triangular portion of the original matrix.
      //    even though the upper triangular elements (excluding diagonal elements)
      //    remain in the original matrix, they are not used in further
      //    computations.  thus, the lower triangular portion of the resulting
      //    matrix is such that A = L*L' where L' denotes the transpose of the
      //    lower triangular matrix L
      // loop over columns of the matrix
      for(size_t k=0; k<A.columns(); ++k){
        A(k,k) = std::sqrt(A(k,k));
        // scale current column
        for(size_t i=k+1; i<A.columns(); ++i)
          A(i,k) = A(i,k)/A(k,k);
        // from each remaining column, subtract multiple of current column
        for(size_t j=k+1; j<A.columns(); ++j)
          for(size_t i=j; i<A.columns(); ++i)
            A(i,j) -= A(i,k)*A(j,k);
      }
      
      // perform forward substitution on the system L*y = b, where y = L'*x and
      // L' denotes the transpose of the lower triangular matrix L stored in A
      // loop over columns
      for(size_t j=0; j<A.columns(); ++j) {
        // stop if matrix is singular
        if(A(j,j) == 0)
          throw std::runtime_error("*** ERROR *** Singular matrix found.");
        // loop over right hand side columns
        for(size_t kk=0; kk<b.columns(); ++kk) {
          // compute solution component
          y[j][kk] = b[j][kk]/A[j][j];
          // loop over rows below the diagonal
          for(size_t i=j+1; i<A.columns(); ++i) {
            // loop over right hand side columns
            // update right hand side
            b[i][kk] -= A(i,j)*y[j][kk];
          }
        }
      }
      
      // perform back substitution on the system L'*x = y to obtain the final
      // soluion vector x
      // loop backwards over columns
      for(int j=(A.columns()-1); j>=0; --j){
        // loop over right hand side columns
        for(size_t kk=0; kk<b.columns(); ++kk) {
          // compute solution component
          x[j][kk] = y[j][kk]/A(j,j);
          // loop over rows above the diagonal
          for(int i=0; i<j; ++i){
            // update right hand side
            y[i][kk] -= x[j][kk]*A(j,i);
          }
        }
      }
      
      // note that the code for back substitution access the lower triangular part
      //    of the matrix
      return x;
    }
    
    
    template <class matrix_type>
    dense_vector<typename matrix_type::value_type> vec(const matrix_type& m) {
      
      typedef typename matrix_type::value_type value_type;
      typedef dense_vector<value_type> vector_type;
      
      vector_type v(m.rows() * m.columns());
      const value_type *array = m.storage();
      for (int i=0; i<v.size(); ++i)
        v[i] = array[i];
      return v;
    }
    
    
    
    //! \relates SparseMatrix Scalar-sparse matrix multiplication operator.
    /*! This operator is used for the multiplication between a scalar and a sparse matrix.
     * \param A - A scalar at the left of the '*' operator.
     * \param B - A constant reference to the sparse matrix at the right of the '*' operator.
     * \return A sparse matrix, result of the multiplication.
     */
    template <typename V>
    inline sparse_matrix<V> operator*(V s, const sparse_matrix<V>& B) {
      sparse_matrix<V> r(B);
      r *= s;
      return r;
    }
    
    //! \relates SparseMatrix Scalar-sparse matrix multiplication operator.
    /*! This operator is used for the multiplication between a scalar and a sparse matrix.
     * \param A - A constant reference to the sparse matrix at the left of the '*' operator.
     * \param B - A scalar at the right of the '*' operator.
     * \return A sparse matrix, result of the multiplication.
     */
    template<typename V>
    inline sparse_matrix<V> operator*(const sparse_matrix<V>& A, V s) {
      sparse_matrix<V> r(A);
      r *= s;
      return r;
    }
    
    
    //! \relates SparseMatrix Sparse matrix - vector multiplication operator.
    /*! This operator is used for the multiplication between a sparse matrix and a full matrix. The
     * evaluation of the resulting matrix is deferred to another object, thus facilitating the Return
     * Value Optimization (RVO).
     * \param A - A constant reference to the sparse matrix at the left of the '*' operator.
     * \param B - A constant reference to the full matrix at the right of the '*' operator.
     * \return A sparse matrix, result of the multiplication.
     */
    template <typename V>
    dense_vector<V> operator*(const sparse_matrix<V>& A, const dense_vector<V>& b) {
      
      typedef typename sparse_matrix<V>::const_hash_iterator const_hash_iterator;
      
      assert(A.columns() == b.size());
      
      dense_vector<V> r(A.rows());
      
      // get number of rows
      size_t m = A.rows();
      
      for (const_hash_iterator it = A.map_.begin(); it != A.map_.end(); ++it) {
        
        // get subscripts
        std::pair<size_t,size_t> subs = A.unhash(it->first);
        r(subs.first) += it->second * b(subs.second);
      }
      return r;
    }
    
    
    //! \relates SparseMatrix Sparse matrix - sparse vector multiplication operator.
    /*! This operator is used for the multiplication between a sparse matrix and a sparse vector.
     * \param A - A constant reference to the sparse matrix at the left of the '*' operator.
     * \param B - A constant reference to the sparse vector at the right of the '*' operator.
     * \return A sparse vector, result of the multiplication.
     */
    template <typename V>
    sparse_vector<V> operator*(const sparse_matrix<V>& A, const sparse_vector<V>& b) {
      
      typedef typename sparse_matrix<V>::const_hash_iterator const_hash_iterator;
      
      assert(A.columns() == b.size());
      
      sparse_vector<V> r(A.rows());
      
      for (const_hash_iterator it = A.map_.begin(); it != A.map_.end(); ++it) {
        
        // get subscripts
        std::pair<size_t,size_t> subs = A.unhash(it->first);
        
        V val = b(subs.second);
        if (val != V())
          r(subs.first) += it->second * b(subs.second);
      }
      return r;
    }
    
    
    __END_CPPUTILS__
    
    
#endif /*CPP_BLAS_HPP_*/
