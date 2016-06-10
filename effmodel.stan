data {
  int D;                // Number of supernovae
  vector[D] ston;
  vector[D] fracincrease;
  int found[D];
}

transformed data {
  vector[D] lnFI;
  real  minlnFI;
  real maxlnFI;
  lnFI <- log(fracincrease);
  print(lnFI);
  minlnFI <- min(lnFI);
  maxlnFI <- max(lnFI);
}
parameters {
  vector<lower=0, upper=1>[2] a_;
  vector<lower=0, upper=10>[2] b_;
  vector<lower=0, upper=10>[2] c_;
}

model {
  vector[D] a;
  vector[D] b;
  vector[D] c;

  vector[D] eff;
  a<- a_[1] + (a_[2]-a_[1])/(maxlnFI-minlnFI)* (lnFI-minlnFI);
  b<- b_[1] + (b_[2]-b_[1])/(maxlnFI-minlnFI)* (lnFI-minlnFI);
  c<- c_[1] + (c_[2]-c_[1])/(maxlnFI-minlnFI)* (lnFI-minlnFI);

  for (d in 1:D) {  
    eff[d] <- tanh((ston[d]-b[d])/c[d]);
    # print(eff[d]);
  }

  eff <- a .* eff;

  found ~ bernoulli(eff);
}