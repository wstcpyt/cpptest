#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_vector.h>
#define PRO_SIZE 20
#define  LAMBDAVALUE 0.1f

class LinearLeast{
    double d {0.25};
public:
    int s;
    gsl_matrix * IdentityLambda;
    gsl_matrix * A1;
    gsl_vector * fexact;
    gsl_vector * x;
    gsl_vector * b;
    gsl_matrix * ATA;
    gsl_matrix * InverseFT;
    gsl_vector * SecondTerm;
    gsl_permutation * perm;
    LinearLeast(){
        initA();
        initb();
        initIdentityLambda();
        addfirstterm();
        invertmatrix();
        initSecondTerm();
        calculatex();
    }
    ~LinearLeast(){
        gsl_matrix_free (A1);
        gsl_matrix_free (ATA);
        gsl_matrix_free (IdentityLambda);
        gsl_matrix_free (InverseFT);
        gsl_permutation_free (perm);
        gsl_vector_free (fexact);
        gsl_vector_free (b);
        gsl_vector_free(SecondTerm);
        gsl_vector_free(x);
    }

private:
    void initA(){
        A1 = gsl_matrix_alloc (PRO_SIZE, PRO_SIZE);
        ATA = gsl_matrix_alloc (PRO_SIZE, PRO_SIZE);
        for (size_t i = 0; i < PRO_SIZE; i++){
            for (int j = 0; j< PRO_SIZE; j++){
                double s = 1.0/PRO_SIZE * (double(j) + 0.5);
                double t = 1.0/PRO_SIZE * (double(i) + 0.5);
                gsl_matrix_set (A1, i, j, 1.0/PRO_SIZE * d/pow(pow(d, 2.0) + pow(s-t, 2.0), 1.5));
            }
        }
        transmulti();
    }
    void transmulti() {
        gsl_blas_dgemm (CblasNoTrans, CblasTrans,
                        1.0, A1, A1,
                        0.0, ATA);
    }

    void initIdentityLambda(){
        IdentityLambda = gsl_matrix_alloc (PRO_SIZE, PRO_SIZE);
        gsl_matrix_set_identity(IdentityLambda);
        gsl_matrix_scale(IdentityLambda, pow(LAMBDAVALUE, 2.0));

    }

    void addfirstterm(){
        gsl_matrix_add(ATA, IdentityLambda);
    }

    void invertmatrix(){
        InverseFT = gsl_matrix_alloc (PRO_SIZE, PRO_SIZE);
        perm = gsl_permutation_alloc (PRO_SIZE);
        // Make LU decomposition of matrix m
        gsl_linalg_LU_decomp (ATA, perm, &s);

        // Invert the matrix m
        gsl_linalg_LU_invert (ATA, perm, InverseFT);
    }

    void initSecondTerm(){
        SecondTerm = gsl_vector_alloc(PRO_SIZE);
        gsl_blas_dgemv(CblasTrans, 1.0, A1, b, 0.0, SecondTerm);
    }

    void initb(){
        b = gsl_vector_alloc(PRO_SIZE);
        fexact = gsl_vector_alloc(PRO_SIZE);
        for (size_t i =0; i< PRO_SIZE; i++){
            if (i<8){
                gsl_vector_set (fexact, i, 2);
            }
            else{
                gsl_vector_set (fexact, i, 1);
            }
        }
        gsl_blas_dgemv(CblasNoTrans, 1.0, A1, fexact, 0.0, b);
    }

    void calculatex(){
        x = gsl_vector_alloc(PRO_SIZE);
        gsl_blas_dgemv(CblasNoTrans, 1.0, InverseFT, SecondTerm, 0.0, x);

    }

};

int
main (void)
{
    LinearLeast linearLeast;
    printf ("[ %g, %g\n", gsl_matrix_get(linearLeast.IdentityLambda ,0, 0), gsl_matrix_get(linearLeast.IdentityLambda ,0, 1));
    printf ("  %g, %g ]\n", gsl_matrix_get(linearLeast.IdentityLambda ,1, 0),gsl_matrix_get(linearLeast.IdentityLambda ,1, 1));


    printf ("[ %g, %g\n", gsl_matrix_get(linearLeast.InverseFT ,0, 0), gsl_matrix_get(linearLeast.InverseFT ,0, 1));
    printf ("  %g, %g ]\n", gsl_matrix_get(linearLeast.InverseFT ,1, 0),gsl_matrix_get(linearLeast.InverseFT ,1, 1));

    printf ("[ %g, %g\n", gsl_vector_get(linearLeast.x ,5), gsl_vector_get(linearLeast.x ,14));



    return 0;
}