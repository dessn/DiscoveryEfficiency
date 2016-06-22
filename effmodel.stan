data {
  int D;                // Number of supernovae
  vector[D] ston;
  vector<lower=0, upper=1>[D] lnFI;
  int found[D];
}

transformed data {
  # vector[D] lnFI;
  vector[D] oneminuslnFI;
  # real  minlnFI_;
  # real maxlnFI_;

  # lnFI <- log(fracincrease);
  # minlnFI_ <- min(lnFI);
  # maxlnFI_ <- max(lnFI);

  # lnFI <- lnFI-minlnFI_;
  # lnFI <- lnFI/(maxlnFI_-minlnFI_);

 
  oneminuslnFI <- 1- lnFI;
}

parameters {
  # vector<lower=0.2, upper=6>[2] a;
  real<lower=0.5, upper=0.8> a1;
  real<lower=3, upper=6> a2;
  vector<lower=1, upper=4>[2] b;
  vector<lower=1, upper=5>[2] c;
  real <lower=3, upper =6> pop;
}

model {

  vector[D] eff;
  vector[D] pterm;

  for (d in 1:D){
    pterm[d] <- oneminuslnFI[d]^a2;
  }

  eff <- (ston-(b[1] + (b[2]-b[1])* pterm)) ./ (c[1] + (c[2]-c[1])* pterm);

  for (d in 1:D) { 
    if (eff[d] <=0){
      eff[d] <- machine_precision();
    } else {
      eff[d] <- (1+(a1-1)*pterm[d]) * tanh(eff[d])^pop;
      # eff[d] <- exp(-a1*((1-lnFI[d])^a2))* tanh(eff[d])^pop;
    }
  }

  found ~ bernoulli(eff);
}