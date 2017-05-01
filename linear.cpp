#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <mpi.h>
#include <set>
#include <map>
#include <time.h>

#ifdef __MACH__
#define CLOCK_REALTIME 0
#define clock_gettime(a,b) 0;
#endif

double eps;
double best_primal;
double* best_w;
enum{L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, L2R_LR_DUAL};
enum{L1, L2LOSS, LR};

static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

static timespec comm_time, comp_time, idle_time, total_comp_time;
static timespec start, end, start1, end1;
static timespec all_start;
timespec io_time;


double global_pos_label;

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};

template<typename T>
void mpi_allreduce(T *buf, const int count, MPI_Datatype type, MPI_Op op)
{
	std::vector<T> buf_reduced(count);

	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&comp_time, timediff(start1,end1));
	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&idle_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Allreduce(buf, buf_reduced.data(), count, type, op, MPI_COMM_WORLD);
	for(int i=0;i<count;i++)
		buf[i] = buf_reduced[i];
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&comm_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME, &end1);
	timeadd(&idle_time, timediff(start1,end1));

	clock_gettime(CLOCK_REALTIME, &start1);
}

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	if(mpi_get_rank()!=0)
		return;
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

static int compute_fun(double* w, double* alpha, double C, int loss, const problem *prob, double stepsize)
{
	static long iter = 0;
	iter++;
	clock_gettime(CLOCK_REALTIME, &end1);
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	timeadd(&comp_time, timediff(start1,end1));
	int i;
	int conv = 0;
	int l = prob->l;
	int w_size = prob->n;
	double reg = 0;
	double lossterm = 0;
	double dual = 0;
	for (i=0;i<w_size;i++)
		reg += w[i]*w[i];

	if (loss == L2LOSS)
		for (i=0;i<l;i++)
			dual += alpha[i] * alpha[i] * 0.25 / C;

	if (loss != LR)
	{
		for (i=0;i<l;i++)
		{
			feature_node const *xi = prob->x[i];
			double d = 1.0 - prob->y[i] * sparse_operator::dot(w, xi);
			if (d > 0)
			{
				if (loss == L2LOSS)
					lossterm += d * d;
				else
					lossterm += d;
			}
			dual -= alpha[i];
		}
	}
	else
		for (i=0;i<l;i++)
		{
			feature_node const *xi = prob->x[i];
			double d = prob->y[i] * sparse_operator::dot(w, xi);
			if (d >= 0)
				lossterm += log(1 + exp(-d));
			else
				lossterm += (-d + log(1 + exp(d)));
			dual += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])	- C * log(C);
		}
	mpi_allreduce_notimer(&dual, 1, MPI_DOUBLE, MPI_SUM);
	mpi_allreduce_notimer(&lossterm, 1, MPI_DOUBLE, MPI_SUM);
	double f = reg / 2.0 + C * lossterm;
	if (f < best_primal)
	{
		best_primal = f;
		memcpy(best_w, w, sizeof(double) * w_size);
	}
	dual = dual + reg / 2.0;
	if ((dual + f)/f < eps)
		conv = 1;

	info("Iter %d Dual %15.20e Primal %15.20e Step %g Time %g\n", iter, dual, f, stepsize, double(total_comp_time.tv_sec) + double(total_comp_time.tv_nsec)/double(1000000000));
	mpi_allreduce_notimer(&conv, 1, MPI_INT, MPI_MAX);
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);
	return conv;
}

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w, int &conv);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int start;
	int length;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	this->C = C;
	int w_size = get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}

double l2r_lr_fun::fun(double *w, int &conv)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;

	Xv(w, z);

	for(i=start;i<start + length;i++)
		f += w[i] * w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}
	mpi_allreduce(&f, 1, MPI_DOUBLE, MPI_SUM);

	clock_gettime(CLOCK_REALTIME, &end1);
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	timeadd(&comp_time, timediff(start1,end1));
	
	int w_size=get_nr_variable();
	info("FUN %15.20e time %g\n",f, double(total_comp_time.tv_sec) + double(total_comp_time.tv_nsec)/double(1000000000));
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);
	for (i=start;i<start + length;i++)
		g[i] = w[i] + g[i];
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int w_size = get_nr_variable();
	int i;
	int l=prob->l;
	double *wa = new double[l];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		wa[i] = sparse_operator::dot(s, xi);
		wa[i] = C[i]*D[i]*wa[i];
		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=start;i<start + length;i++)
		Hs[i] = s[i] + Hs[i];
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w, int &conv);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int start;
	int length;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = C;
	int w_size = get_nr_variable();
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	this->start = shift * rank;
	this->length = min(max(w_size - start, 0), shift);
	if (length == 0)
		start = 0;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w, int &conv)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=start;i<start + length;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}
	mpi_allreduce(&f, 1, MPI_DOUBLE, MPI_SUM);

	clock_gettime(CLOCK_REALTIME, &end1);
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	timeadd(&comp_time, timediff(start1,end1));
	
	info("FUN %15.20e time %g\n",f, double(total_comp_time.tv_sec) + double(total_comp_time.tv_nsec)/double(1000000000));
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = 2 * C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);
	for(i=start;i<start + length;i++)
		g[i] = w[i] + g[i];
	mpi_allreduce(g, w_size, MPI_DOUBLE, MPI_SUM);
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);

		wa[i] = 2 * C[I[i]]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=start;i<start + length;i++)
		Hs[i] = s[i] + Hs[i];
	mpi_allreduce(Hs, w_size, MPI_DOUBLE, MPI_SUM);
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}

#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual_disdca(const problem *prob, double *w, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	int start_count = shift * rank;
	double delta = 0;
	int length = min(max(w_size - start_count, 0), shift);
	if (length == 0)
		start_count = 0;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 10; // for inner Newton
	int max_local_iter = 1;
	double upper_bound[3] = {Cn, 0, Cp};


	for(i=0; i<l; i++)
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;

	
	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = nr_node * sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}
	int converged = 0;
	double logconst = 0;
	for (i=0;i<l;i++)
		logconst -= upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	mpi_allreduce_notimer(&logconst, 1, MPI_DOUBLE, MPI_SUM);
	clock_gettime(CLOCK_REALTIME, &start1);
	clock_gettime(CLOCK_REALTIME, &all_start);

	while (iter < max_iter && converged == 0)
	{
		iter++;

		for (int local_iter = 0;local_iter < max_local_iter; local_iter++)
		{
			for (i=0; i<l; i++)
			{
				int j = i+rand()%(l-i);
				swap(index[i], index[j]);
			}
			int newton_iter = 0;
			double Gmax = 0;
			for (s=0; s<l; s++)
			{
				i = index[s];
				const schar yi = y[i];
				double C = upper_bound[GETI(i)];
				double ywTx = 0, xisq = xTx[i];
				feature_node * const xi = prob->x[i];
				ywTx = yi*sparse_operator::dot(w, xi);
				double a = xisq, b = ywTx;

				// Decide to minimize g_1(z) or g_2(z)
				int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
				if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
				{
					ind1 = 2*i+1;
					ind2 = 2*i;
					sign = -1;
				}

				//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
				double alpha_old = alpha[ind1];
				double z = alpha_old;
				if(C - z < 0.5 * C)
					z = 0.1*z;
				double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				Gmax = max(Gmax, fabs(gp));

				// Newton method on the sub-problem
				const double eta = 0.1; // xi in the paper
				int inner_iter = 0;
				while (inner_iter <= max_inner_iter)
				{
					double gpp = a + C/(C-z)/z;
					double tmpz = z - gp/gpp;
					if(tmpz <= 0)
						z *= eta;
					else // tmpz in (0, C)
						z = tmpz;
					gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
					inner_iter++;
				}

				if(inner_iter > 0) // update w
				{
					alpha[ind1] = z;
					alpha[ind2] = C-z;
					sparse_operator::axpy(sign*(z-alpha_old)*yi*nr_node, xi, w);
				}
			}
		}

		mpi_allreduce(w, w_size, MPI_DOUBLE, MPI_SUM);
		for (i=0;i<w_size;i++)
			w[i] /= nr_node;

		converged += compute_fun(w, alpha, Cp, LR, prob, 1.0);
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;

}
#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double Cp, double Cn, double beta)
{
	int l = prob->l;
	int w_size = prob->n;
	int nr_node = mpi_get_size();
	int rank = mpi_get_rank();
	int shift = int(ceil(double(w_size) / double(nr_node)));
	int start_count = shift * rank;
	double delta = 0;
	int length = min(max(w_size - start_count, 0), shift);
	if (length == 0)
		start_count = 0;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];	
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 10; // for inner Newton
	int max_local_iter = 1;
	double upper_bound[3] = {Cn, 0, Cp};
	double *alpha_orig = new double[2 * l];
	double *alpha_inc = new double[2 * l];
	double *w_orig = new double[w_size];
	double buffer2[3] = {0,0,0};
	double oldlogterm = 0;
	double *allreduce_buffer = new double[w_size + 2];
	double sigma = 0.01;
	double w_inc_square;
	double w_dot_w_inc;
	double logterm;
	double dual_obj_inc;
	double theta;


	for(i=0; i<l; i++)
		if(prob->y[i] > 0)
			y[i] = +1;
		else
			y[i] = -1;

	
	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
		oldlogterm += alpha[2 * i] * log(alpha[2 * i]) + alpha[2 * i + 1] * log(alpha[2 * i + 1]);
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		xTx[i] = sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[2*i], xi, w);
		index[i] = i;
	}
	int converged = 0;
	double logconst = 0;
	for (i=0;i<l;i++)
		logconst -= upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	mpi_allreduce_notimer(&logconst, 1, MPI_DOUBLE, MPI_SUM);
	clock_gettime(CLOCK_REALTIME, &start1);
	clock_gettime(CLOCK_REALTIME, &all_start);

	while (iter < max_iter && converged == 0)
	{
		iter++;
		memcpy(w_orig, w, sizeof(double) * w_size);
		memcpy(alpha_orig, alpha, sizeof(double) * 2 * l);
		memset(alpha_inc, 0, sizeof(double) * 2 * l);
		w_inc_square = 0;
		w_dot_w_inc = 0;
		logterm = 0;

		for (int local_iter = 0;local_iter < max_local_iter; local_iter++)
		{
			for (i=0; i<l; i++)
			{
				int j = i+rand()%(l-i);
				swap(index[i], index[j]);
			}
			int newton_iter = 0;
			double Gmax = 0;
			for (s=0; s<l; s++)
			{
				i = index[s];
				const schar yi = y[i];
				double C = upper_bound[GETI(i)];
				double ywTx = 0, xisq = xTx[i];
				feature_node * const xi = prob->x[i];
				ywTx = yi*sparse_operator::dot(w, xi);
				double a = xisq, b = ywTx;

				// Decide to minimize g_1(z) or g_2(z)
				int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
				if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
				{
					ind1 = 2*i+1;
					ind2 = 2*i;
					sign = -1;
				}

				//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
				double alpha_old = alpha[ind1];
				double z = alpha_old;
				if(C - z < 0.5 * C)
					z = 0.1*z;
				double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				Gmax = max(Gmax, fabs(gp));

				// Newton method on the sub-problem
				const double eta = 0.1; // xi in the paper
				int inner_iter = 0;
				while (inner_iter <= max_inner_iter)
				{
					double gpp = a + C/(C-z)/z;
					double tmpz = z - gp/gpp;
					if(tmpz <= 0)
						z *= eta;
					else // tmpz in (0, C)
						z = tmpz;
					gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
					inner_iter++;
				}

				if(inner_iter > 0) // update w
				{
					alpha[ind1] = z;
					alpha[ind2] = C-z;
					sparse_operator::axpy(sign*(z-alpha_old)*yi, xi, w);
				}
			}
		}

		for (i=0;i<2 * l;i++)
			alpha_inc[i] = alpha[i] - alpha_orig[i];
		theta = 1;
		int back_iter = 0;
		for (i=0;i<w_size;i++)
			allreduce_buffer[i] = w[i] - w_orig[i];

		if (beta > 0)
		{
			logterm = 0;
			double logterm_for_delta = 0;
			for (i=0;i<2 * l;i++)
			{
				double tmp = alpha[i];
				logterm += tmp * log(tmp);
			}
			logterm_for_delta = logterm;
			allreduce_buffer[w_size] = logterm;
			mpi_allreduce(allreduce_buffer, w_size + 1, MPI_DOUBLE, MPI_SUM);
			logterm = allreduce_buffer[w_size];
			buffer2[0] = 0;
			buffer2[1] = 0;
			for (i=start_count;i<start_count + length;i++)
			{
				buffer2[0] += w_orig[i] * allreduce_buffer[i];
				buffer2[1] += allreduce_buffer[i] * allreduce_buffer[i];
			}
			mpi_allreduce(buffer2, 2, MPI_DOUBLE, MPI_SUM);
			delta = buffer2[0] + logterm - oldlogterm;
			w_dot_w_inc = buffer2[0];
			w_inc_square = buffer2[1] * 0.5;
			dual_obj_inc = logterm - oldlogterm + w_dot_w_inc + w_inc_square;

			while (dual_obj_inc > theta * sigma * delta && back_iter < 500)
			{
				theta *= beta;
				back_iter++;
				logterm = 0;
				for (i=0;i<2 * l;i++)
				{
					double tmp = alpha_orig[i] + theta * alpha_inc[i];
					logterm += tmp * log(tmp);
				}
				mpi_allreduce(&logterm, 1, MPI_DOUBLE, MPI_SUM);
				dual_obj_inc = logterm - oldlogterm + w_dot_w_inc * theta + w_inc_square * theta * theta;
			}
		}
		else
		{
			mpi_allreduce(allreduce_buffer, w_size, MPI_DOUBLE, MPI_SUM);
			theta = 1.0 / nr_node;
		}
		if (back_iter >= 500)
		{
			info("WARNING: Backtracking failed\n");
			memcpy(w, w_orig, sizeof(double) * w_size);
			memcpy(alpha, alpha_orig, sizeof(double) * 2 * l);
			break;
		}


		if ((beta > 0 && back_iter > 0) || beta <= 0)
		{
			for (i=0;i<w_size;i++)
				w[i] = w_orig[i] + theta * allreduce_buffer[i];
			for (i=0;i<2 * l;i++)
				alpha[i] = alpha_orig[i] + theta * alpha_inc[i];
		}
		else
			for (i=0;i<w_size;i++)
				w[i] = w_orig[i] + allreduce_buffer[i];
		oldlogterm = logterm;
		converged += compute_fun(w, alpha, Cp, LR, prob, theta);
	}

	info("\noptimization finished, #iter = %d\n",iter);

	// calculate objective value

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
	delete [] w_orig;
	delete [] alpha_orig;
	delete [] alpha_inc;
	delete [] allreduce_buffer;

}
#undef GETI

#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)
static void solve_l2r_l1l2_svc(
	const problem *prob, double *w,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int size = mpi_get_size();
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 100000;
	int max_inner_iter;
	int *index = new int[l];
	double *alpha = new double[l];
	double *alpha_orig = new double[l];
	double *alpha_inc = new double[l];
	double *w_orig = new double[w_size];
	double *allreduce_buffer;
	double lambda = 0;
	schar *y = new schar[l];
	int converged = 0;
	int out_iter = 0;
	double eta = 1;
	int L2 = 1;
	double sigma = 0.1;
	double back_track = 0.5;
	double one_over_log_back = 1/log(0.5);
	double log_two = log(2.0);
	double max_step;
	int reduce_length = w_size;
	max_inner_iter = 1;
	reduce_length += 3;
	allreduce_buffer = new double[reduce_length];

	// PG: projected gradient, for shrinking and stopping
	double PG;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		L2 = 0;
		lambda = 1e-3;
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = diag[GETI(i)] + lambda + sparse_operator::nrm2_sq(xi);
		sparse_operator::axpy(y[i]*alpha[i], xi, w);
		index[i] = i;
	}

	while (!converged && out_iter < max_iter)
	{
		out_iter++;
		iter = 0;
		for (i=0;i<l;i++)
			alpha_orig[i] = alpha[i];
		for (i=0;i<w_size;i++)
			w_orig[i] = w[i];
		for (int inner = 0;inner<max_inner_iter;inner++)
		{
			for (i=0; i<l; i++)
			{
				int j = i+rand()%(l - i);
				swap(index[i], index[j]);
			}


			for (s=0; s<l; s++)
			{
				i = index[s];
				schar yi = y[i];

				feature_node * const xi = prob->x[i];
				G = yi*sparse_operator::dot(w, xi) - 1;
				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];

				PG = 0;
				if (alpha[i] == 0)
				{
					if (G < 0)
						PG = G;
				}
				else if (alpha[i] == C)
				{
					if (G > 0)
						PG = G;
				}
				else
					PG = G;

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = (alpha[i] - alpha_old)*yi;
					sparse_operator::axpy(d, xi, w);
				}
			}

			iter++;
		}

		for (i=0;i<l;i++)
			alpha_inc[i] = alpha[i] - alpha_orig[i];
		for (i=0;i<w_size;i++)
			allreduce_buffer[i] = w[i] - w_orig[i];
		max_step = INF;
		double sum_alpha_inc = 0;
		double alpha_square = 0;
		double alpha_square_alpha = 0;
		for (i=0;i<l;i++)
		{
			sum_alpha_inc += alpha_inc[i];
			alpha_square += alpha_inc[i]*alpha_inc[i]*diag[GETI(i)];
			alpha_square_alpha += alpha_inc[i]*alpha_orig[i]*diag[GETI(i)];
			if (alpha_inc[i] > 0)
				max_step = min(max_step, (upper_bound[GETI(i)] - alpha_orig[i]) / alpha_inc[i]);
			else if (alpha_inc[i] < 0)
				max_step = min(max_step, alpha_orig[i] / (-alpha_inc[i]));
		}
		allreduce_buffer[w_size] = sum_alpha_inc;
		allreduce_buffer[w_size + 1] = alpha_square;
		allreduce_buffer[w_size + 2] = alpha_square_alpha;
		mpi_allreduce(&max_step, 1, MPI_DOUBLE, MPI_MIN);

		mpi_allreduce(allreduce_buffer, reduce_length, MPI_DOUBLE, MPI_SUM);

		double w_inc_square;
		double w_w_inc;
		w_w_inc = 0;
		w_inc_square = 0;
		for (i=0;i<w_size;i++)
		{
			w_inc_square += allreduce_buffer[i]*allreduce_buffer[i];
			w_w_inc += allreduce_buffer[i] * w_orig[i];
		}

		double opt = (allreduce_buffer[w_size] - allreduce_buffer[w_size+2] - w_w_inc)/(allreduce_buffer[w_size+1] + w_inc_square);
		eta = min(opt, max_step);

		if (eta <= 0)
		{
			info("WARNING: negative step size\n");
			memcpy(w, w_orig, sizeof(double) * w_size);
			memcpy(alpha, alpha_orig, sizeof(double) * 2 * l);
			break;
		}

		for (i=0;i<w_size;i++)
			w[i] = w_orig[i] + eta * allreduce_buffer[i];
		for (i=0;i<l;i++)
			alpha[i] = alpha_orig[i] + eta * alpha_inc[i];

		converged += compute_fun(w, alpha, Cp, L2, prob, eta);
		clock_gettime(CLOCK_REALTIME, &start1);
		clock_gettime(CLOCK_REALTIME, &all_start);
	}

	info("\noptimization finished, #iter = %d\n",out_iter);

	int nSV = 0;
	for(i=0; i<l; i++)
		if(alpha[i] > 0)
			++nSV;
	mpi_allreduce_notimer(&nSV, 1, MPI_INT, MPI_SUM);

	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
	delete [] alpha_inc;
	delete [] alpha_orig;
	delete [] w_orig;
}

static void solve_l2r_l1l2_disdca(
	const problem *prob, double *w,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int size = mpi_get_size();
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 100000;
	int max_inner_iter;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int converged = 0;
	int out_iter = 0;
	int L2 = 1;
	max_inner_iter = 1;

	// PG: projected gradient, for shrinking and stopping
	double PG;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L1_DISDCA)
	{
		L2 = 0;
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node * const xi = prob->x[i];
		QD[i] = size * sparse_operator::nrm2_sq(xi) + diag[GETI(i)];
		sparse_operator::axpy(y[i]*alpha[i], xi, w);
		index[i] = i;
	}

	while (!converged && out_iter < max_iter)
	{
		out_iter++;
		iter = 0;
		for (int inner = 0;inner<max_inner_iter;inner++)
		{
			for (i=0; i<l; i++)
			{
				int j = i+rand()%(l - i);
				swap(index[i], index[j]);
			}


			for (s=0; s<l; s++)
			{
				i = index[s];
				schar yi = y[i];

				feature_node * const xi = prob->x[i];
				G = yi*sparse_operator::dot(w, xi) - 1;
				C = upper_bound[GETI(i)];
				G += alpha[i]*diag[GETI(i)];

				PG = 0;
				if (alpha[i] == 0)
				{
					if (G < 0)
						PG = G;
				}
				else if (alpha[i] == C)
				{
					if (G > 0)
						PG = G;
				}
				else
					PG = G;

				if(fabs(PG) > 1.0e-12)
				{
					double alpha_old = alpha[i];
					alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
					d = (alpha[i] - alpha_old)*yi*size;
					sparse_operator::axpy(d, xi, w);
				}
			}

			iter++;
		}

		mpi_allreduce(w, w_size, MPI_DOUBLE, MPI_SUM);
		for (i=0;i<w_size;i++)
			w[i] /= size;

		converged += compute_fun(w, alpha, Cp, L2, prob, 1.0);
		clock_gettime(CLOCK_REALTIME, &start1);
		clock_gettime(CLOCK_REALTIME, &all_start);
	}

	info("\noptimization finished, #iter = %d\n",out_iter);

	int nSV = 0;
	for(i=0; i<l; i++)
		if(alpha[i] > 0)
			++nSV;
	mpi_allreduce_notimer(&nSV, 1, MPI_INT, MPI_SUM);

	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

#undef GETI

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
//
// In distributed environment, we need to make sure that the order of labels
// are consistent. It is achieved by three steps. Please see the comments in
// ``group_classes.''
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int i;


	// Step 1. Each node collects its own labels.
	// If you have enormous number of labels, you can use std::unordered_set
	// (whose complexity is O(1)) to replace std::set (whose complexity is
	// O(log(n))). Because std::unordered_set needs a compiler supporting C++11,
	// we use std::set for a better compatibility. Similarly, you may want to
	// replace std::map with std::unordered_map.
	std::set<int> label_set;
	for(i=0;i<prob->l;i++)
		label_set.insert((int)prob->y[i]);

	// Step 2. All labels are sent to the first machine.
	if(mpi_get_rank()==0)
	{
		for(i=1;i<mpi_get_size();i++)
		{
			MPI_Status status;
			int size;
			MPI_Recv(&size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			std::vector<int> label_buff(size);
			MPI_Recv(label_buff.data(), size, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			for(int j=0; j<size; j++)
				label_set.insert(label_buff[j]);
		}
	}
	else
	{
		int size = (int)label_set.size();
		std::vector<int> label_buff(size);
		i = 0;
		for(std::set<int>::iterator this_label=label_set.begin();
				this_label!=label_set.end(); this_label++)
		{
			label_buff[i] = (*this_label);
			i++;
		}
		MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(label_buff.data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	// Step 3. The first machine broadcasts the global labels to other nodes, so that
	// the order of labels in each machine is consistent.
	int nr_class = (int)label_set.size();
	MPI_Bcast(&nr_class, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::map<int, int> label_map;
	int *label = Malloc(int, nr_class);
	{
		if(mpi_get_rank()==0)
		{
			i = 0;
			for(std::set<int>::iterator this_label=label_set.begin();
					this_label!=label_set.end(); this_label++)
			{
				label[i] = (*this_label);
				i++;
			}
		}
		MPI_Bcast(label, nr_class, MPI_INT, 0, MPI_COMM_WORLD);
		for(i=0;i<nr_class;i++)
			label_map[label[i]] = i;
	}

	// The following codes are similar to the original LIBLINEAR
	int *start = Malloc(int, nr_class);
	int *count = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
		count[i] = 0;
	for(i=0;i<prob->l;i++)
		count[label_map[(int)prob->y[i]]]++;

	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[label_map[(int)prob->y[i]]]] = i;
		++start[label_map[(int)prob->y[i]]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double runtime[4];
	clock_gettime(CLOCK_REALTIME, &all_start);
	clock_gettime(CLOCK_REALTIME, &start1);
	total_comp_time = timediff(start,start);
	comp_time = timediff(start1,start1);
	idle_time = timediff(start1,start1);
	comm_time = timediff(start1,start1);
	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2_TRON:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			double local_eps;
			local_eps = eps;
			TRON tron_obj(fun_obj, local_eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			clock_gettime(CLOCK_REALTIME, &end1);
			timeadd(&comp_time, timediff(start1,end1));

			break;
		}
		case LR_TRON:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			double local_eps;
			local_eps = eps;
			TRON tron_obj(fun_obj, local_eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			clock_gettime(CLOCK_REALTIME, &end1);
			timeadd(&comp_time, timediff(start1,end1));
			break;
		}
		case L2_BDA:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L1_BDA:
			solve_l2r_l1l2_svc(prob, w, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case LR_BDA:
			solve_l2r_lr_dual(prob, w, Cp, Cn, 0.5);
			break;
		case LR_DISDCA:
			solve_l2r_lr_dual_disdca(prob, w, Cp, Cn);
			break;
		case L2_DISDCA:
			solve_l2r_l1l2_disdca(prob, w, Cp, Cn, L2_DISDCA);
			break;
		case L1_DISDCA:
			solve_l2r_l1l2_disdca(prob, w, Cp, Cn, L1_DISDCA);
			break;
		default:
		{
			if(mpi_get_rank() == 0)
				fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
		}
	}
	clock_gettime(CLOCK_REALTIME, &end);
	timeadd(&total_comp_time, timediff(all_start, end));
	runtime[0] = double(comp_time.tv_sec) + double(comp_time.tv_nsec)/double(1000000000);
	runtime[1] = double(comm_time.tv_sec) + double(comm_time.tv_nsec)/double(1000000000);
	runtime[2] = double(idle_time.tv_sec) + double(idle_time.tv_nsec)/double(1000000000);
	runtime[3] = double(io_time.tv_sec) + double(io_time.tv_nsec)/double(1000000000);
	mpi_allreduce_notimer(runtime, 4, MPI_DOUBLE, MPI_SUM);
	for (int i=0;i<4;i++)
		runtime[i] /= mpi_get_size();
	info("Computation Time: %g s, Sync Time: %g s, Communication Time: %g s, IO Time: %g s\n", runtime[0], runtime[2], runtime[1], runtime[3]);
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);
	eps = param->eps;
	best_w = new double[n];
	best_primal = INF;

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);
				int e0 = start[0]+count[0];
				k=0;
				global_pos_label = label[0];
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];
					global_pos_label = label[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int j=0;j<w_size;j++)
						model_->w[j*nr_class+i] = w[j];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	free(best_w);
	return model_;
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}


static const char *solver_type_table[]=
{
"L2_BDA", "L1_BDA", "L2_TRON", "LR_TRON", "L2_DISDCA", "L1_DISDCA", "LR_DISDCA",NULL}; /* solver_type */

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"[rank %d] unknown solver type.\n", mpi_get_rank());

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"[rank %d] unknown text in model file: [%s]\n",mpi_get_rank(),cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2_BDA && param->solver_type != L1_BDA && param->solver_type != LR_BDA && param->solver_type != L2_TRON && param->solver_type != L2_DISDCA && param->solver_type != L1_DISDCA && param->solver_type != LR_TRON && param->solver_type != LR_DISDCA)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}


int mpi_get_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;	
}

int mpi_get_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;	
}

void mpi_exit(const int status)
{
	MPI_Finalize();
	exit(status);
}
