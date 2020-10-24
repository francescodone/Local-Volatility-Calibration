#if 0
// this is a functional definition of tridag
fun *[real] tridagPar( [real] a, *[real] b, [real] c, *[real] y ) =
    let n    = size(0, a) in
    ----------------------------------------------------
    -- Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    --   solved by scan with 2x2 matrix mult operator --
    ----------------------------------------------------
    let b0   = b[0] in
    let mats = map ( fn {real,real,real,real} (int i) =>
		         if 0 < i 
			 then {b[i], 0.0-a[i]*c[i-1], 1.0, 0.0}
			 else {1.0,  0.0,             0.0, 1.0}
		   , iota(n) ) in
    let scmt = scan( fn {real,real,real,real} ( {real,real,real,real} a, 
					        {real,real,real,real} b ) =>
		         let {a0,a1,a2,a3} = a   in
			 let {b0,b1,b2,b3} = b   in
			 let val = 1.0/(a0*b0)   in
			 { (b0*a0 + b1*a2)*val,
			   (b0*a1 + b1*a3)*val,
			   (b2*a0 + b3*a2)*val,
			   (b2*a1 + b3*a3)*val 
			 }
		   , {1.0,  0.0, 0.0, 1.0}, mats ) in
    let b    = map ( fn real ({real,real,real,real} tup) =>
		         let {t0,t1,t2,t3} = tup in
			 (t0*b0 + t1) / (t2*b0 + t3)
		   , scmt ) in
    ------------------------------------------------------
    -- Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    --   solved by scan with linear func comp operator  --
    ------------------------------------------------------
    let y0   = y[0] in
    let lfuns= map ( fn {real,real} (int i) =>
		         if 0 < i 
		 	 then {y[i], 0.0-a[i]/b[i-1]}
			 else {0.0,  1.0            }
		   , iota(n) ) in
    let cfuns= scan( fn {real,real} ({real,real} a, {real,real} b) =>
		         let {a0,a1} = a in
			 let {b0,b1} = b in
			 { b0 + b1*a0, a1*b1 }
		   , {0.0, 1.0}, lfuns ) in
    let y    = map ( fn real ({real,real} tup) =>
		         let {a,b} = tup in
			 a + b*y0
		   , cfuns ) in
    ------------------------------------------------------
    -- Recurrence 3: backward recurrence solved via     --
    --             scan with linear func comp operator  --
    ------------------------------------------------------
    let yn   = y[n-1]/b[n-1] in
    let lfuns= map ( fn {real,real} (int k) =>
		         let i = n-k-1
			 in  if   0 < k
		 	     then {y[i]/b[i], 0.0-c[i]/b[i]}
			     else {0.0,       1.0          }
		   , iota(n) ) in
    let cfuns= scan( fn {real,real} ({real,real} a, {real,real} b) =>
		         let {a0,a1} = a in
			 let {b0,b1} = b in
			 {b0 + b1*a0, a1*b1}
		   , {0.0, 1.0}, lfuns ) in
    let y    = map ( fn real ({real,real} tup) =>
		         let {a,b} = tup in
			 a + b*yn
		   , cfuns ) in
    let y    = map (fn real (int i) => y[n-i-1], iota(n)) in
    copy(y)
#endif

struct MyReal4 {
    REAL x;
    REAL y;
    REAL z;
    REAL w;
    
    // constructors
    inline MyReal4() { x = y = z = w = 0.0; }
    inline MyReal4(const REAL a, const REAL b, const REAL c, const REAL d) {
        x = a; y = b; z = c; w = d;
    }
    // copy constructor
    inline MyReal4(const MyReal4& i4) { 
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
    }
    // assignment operator
    inline MyReal4& operator=(const MyReal4& i4) {
        x = i4.x; y = i4.y; z = i4.z; w = i4.w; 
        return *this;
    }
};

struct MatMult2b2 {
  typedef MyReal4 OpTp;
  static MyReal4 apply(const MyReal4 a, const MyReal4 b) {
    REAL val = 1.0/(a.x*b.x);
    return MyReal4( (b.x*a.x + b.y*a.z)*val,
                    (b.x*a.y + b.y*a.w)*val,
                    (b.z*a.x + b.w*a.z)*val,
                    (b.z*a.y + b.w*a.w)*val );
  }
};

struct MyReal2 {
    REAL x;
    REAL y;
    // constructors
    inline MyReal2() { x = y = 0.0; }
    inline MyReal2(const REAL a, const REAL b) {
        x = a; y = b;
    }
    // copy constructor
    inline MyReal2(const MyReal2& i4) { 
        x = i4.x; y = i4.y; 
    }
    // assignment operator
    inline MyReal2& operator=(const MyReal2& i4) {
        x = i4.x; y = i4.y; 
        return *this;
    }
};

struct LinFunComp {
  typedef MyReal2 OpTp;
  static MyReal2 apply(const MyReal2 a, const MyReal2 b) {
    return MyReal2( b.x + b.y*a.x, a.y*b.y );
  }
};

template<class OP>
void inplaceScanInc(const int n, vector<typename OP::OpTp>& inpres) {
  typename OP::OpTp acc = inpres[0];
  for(int i=1; i<n; i++) {
    acc = OP::apply(acc,inpres[i]);
    inpres[i] = acc;    
  }
}

inline void tridagPar(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
) {
    int i, offset;

    //vector<MyReal4> scanres(n); // supposed to also be in shared memory and to reuse the space of mats
    //--------------------------------------------------
    // Recurrence 1: b[i] = b[i] - a[i]*c[i-1]/b[i-1] --
    //   solved by scan with 2x2 matrix mult operator --
    //--------------------------------------------------
    vector<MyReal4> mats(n);    // supposed to be in shared memory!
    REAL b0 = b[0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { mats[i].x = 1.0;  mats[i].y = 0.0;          mats[i].z = 0.0; mats[i].w = 1.0; }
        else      { mats[i].x = b[i]; mats[i].y = -a[i]*c[i-1]; mats[i].z = 1.0; mats[i].w = 0.0; }
    }
    inplaceScanInc<MatMult2b2>(n,mats);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        uu[i] = (mats[i].x*b0 + mats[i].y) / (mats[i].z*b0 + mats[i].w);
    }
    // b -> uu
    //----------------------------------------------------
    // Recurrence 2: y[i] = y[i] - (a[i]/b[i-1])*y[i-1] --
    //   solved by scan with linear func comp operator  --
    //----------------------------------------------------
    vector<MyReal2> lfuns(n);
    REAL y0 = r[0];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = r[i]; lfuns[i].y = -a[i]/uu[i-1]; }
    }
    inplaceScanInc<LinFunComp>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        u[i] = lfuns[i].x + y0*lfuns[i].y;
    }
    // y -> u

    //----------------------------------------------------
    // Recurrence 3: backward recurrence solved via     --
    //             scan with linear func comp operator  --
    //----------------------------------------------------
    REAL yn = u[n-1]/uu[n-1];
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        int k = n - i - 1;
        if (i==0) { lfuns[0].x = 0.0;  lfuns[0].y = 1.0;           }
        else      { lfuns[i].x = u[k]/uu[k]; lfuns[i].y = -c[k]/uu[k]; }
    }
    inplaceScanInc<LinFunComp>(n,lfuns);
    for(int i=0; i<n; i++) { //parallel, map-like semantics
        u[n-i-1] = lfuns[i].x + yn*lfuns[i].y;
    }
}

