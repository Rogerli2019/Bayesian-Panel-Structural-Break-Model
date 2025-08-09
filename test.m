% clear a0ll 
clear all;
clc;

% set seed
rng(2023,'twister');

T = 100;
N = 25;

% error break at 50
sig2_1 = 0.25 * rand(1,N);
% sig2_2 = sig2_1;
sig2_2 = 0.5 + 0.25 * rand(1,N);
% sig2_1 = [0*ones(1,30)];
% sig2_2 = [0*ones(1,30)];

% beta break at 40,65(do not assume intercept change)
beta1 = 1 * rand(2,N); % first break before 40
beta_change = 1 * rand(2,N);
beta2 = beta1 + 1 * beta_change; % second break between 40 and 65
beta3 = beta1 + 2 * beta_change;  % third break after 65

%beta1 = [0.1*ones(1,15),0.2*ones(1,15);0.1*ones(1,15),0.2*ones(1,15)];
%beta2 = [0.4*ones(1,15),0.5*ones(1,15);0.4*ones(1,15),0.5*ones(1,15)];
%beta3 = [0.8*ones(1,15),0.9*ones(1,15);0.8*ones(1,15),0.9*ones(1,15)];

% covariate
x = cat(2,ones(T,1,N),randn(T,1,N)); % 100 * 2 * 30
% x = cat(2,ones(T,1,N),randn(T,1,N));

% generate y 
y = zeros(T,N);

e = randn(T,N);

for t = 1:T
    x_t = squeeze(x(t,:,:));
    if t < 41
        y(t,:) = beta1(1,:).* x_t(1,:) + beta1(2,:).* x_t(2,:) + sqrt(sig2_1).*e(t,:);
    elseif t<51
        y(t,:) = beta2(1,:).* x_t(1,:) + beta2(2,:).* x_t(2,:) + sqrt(sig2_1).*e(t,:);
    elseif t<66
        y(t,:) = beta2(1,:).* x_t(1,:) + beta2(2,:).* x_t(2,:) + sqrt(sig2_2).*e(t,:);
    else
        y(t,:) = beta3(1,:).* x_t(1,:) + beta3(2,:).* x_t(2,:) + sqrt(sig2_2).*e(t,:);
    end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%         Rolling Regression                          %%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tt = 20;
% 
% beta_it = zeros(2,T-tt,N);
% 
% for it = 1:T-tt
%     for iyn = 1:N
%         xx = squeeze(x(it:it+tt,:,iyn));
%         yy = squeeze(y(it:it+tt,iyn));
%         beta_it(:,it,iyn) = (xx'*xx)\xx'*yy;
%     end
% end
% 
% hold on
% for i = 1:N
%     plot(1:(T-tt), beta_it(2,:,i));
%     ylim([0 2]);
% end
% hold off
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
beta_true = zeros(N,T);
for t = 1:T
    if t < 41
        beta_true(:,t) = beta1(2,:)';
    elseif t<66
        beta_true(:,t) = beta2(2,:)';
    else
        beta_true(:,t) = beta3(2,:)';
    end
end

sig2_true = zeros(N,T);
for t = 1:T
    if t < 50
        sig2_true(:,t) = sig2_1;
    else
        sig2_true(:,t) = sig2_2;
    end
end
% 
y = reshape(y,[T,1,N]);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set prior parameter - follow Smith(2017)
a = 2; b = 2; % std - Smith(2017)
c = 2; d = 0.08; % Every 25 period

% set the prior pertub parameter s - tuned to let alpha_pert equal to 0.25
% current alpha_pert greater than 0.5
s0 = 1;
sig2beta = 0.5;
% mu0 = [0,0];

% matrix version - for theta
kai = 2; % number of coefficients
v0 = eye(kai)*sig2beta;

% set MC parameter
nsim = 2000;burnin = 500;
thining = 3;

k0 = 3;% start with 3 break point(except two end points)
regime_k = k0 + 1; % means 4 regimes
tau = sort(randsample((T-1),k0))'; % generate break point
tau = [40,50,65];
tau = [0,tau,T];
% tau = [0,40,50,65,T];

nsave = (nsim-burnin)/thining;
% storage value
% last dimension is number of simulation
store_sig2 = zeros(N,T,nsave);
store_beta = zeros(N,2,T,nsave);
store_k = zeros(1,nsave);
store_tau = zeros(T,nsave);
store_alpha_pert = zeros(1,nsave);

%  value in one simulation of one regime
% last dimension is number of portfolios
Sig_inv = zeros(kai,kai,N);
Sig_i = zeros(kai,kai);
rho_ik = zeros(kai,N);
b_ik = zeros(1,N); 

% store value in the pertube step/birth/death - 2 new periods
% only use the first value in reverse jump procedure
% last number is 2 periods
stemp_Sig = zeros(2,2,N,2);
stemp_a = zeros(1,2);
stemp_b = zeros(N,2);
stemp_sig2 = zeros(N,2);
stemp_beta = zeros(N,2,2);

% calculate before the loop to save time
inv_v0 = v0\speye(2); % faster 
% mu0sig0mu0 = mu0*inv_v0*mu0';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MCMC procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for isim = 1:nsim
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 1.initial draw
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num_regime = length(tau)-1; % breakpoint + 2 end point - 1  
    % number of regime or number of break point
    K = num_regime - 1; % number of break/number break point except 2 points is less 1
    %% store value in every REGIME used in break point, change with number of regime
    % last dimension is number of regime
    temp_Sig = zeros(2,2,N,num_regime);
    temp_a = zeros(1,num_regime);
    temp_b = zeros(N,num_regime);
    % store value in every REGIME
    temp_sig2 = zeros(N,num_regime);
    temp_beta = zeros(N,2,num_regime);
    temp_l = zeros(1,num_regime);
    %% first generate the initial draw, given tau
    for irgm = 1:num_regime % each regime
        % get the regime length and the correpoding regime
        l_k = tau(irgm+1) - tau(irgm);
        rgm_k = (tau(irgm)+1):tau(irgm+1);
        temp_l(irgm) = l_k;

        % loop over the regime
        a_k = a + l_k/2; % a_k is same to all portfolio
        % store temp a_k - same for every portfolio
        temp_a(irgm) = a_k;% a_k is the same to every value

        for i = 1:N
            xi = squeeze(x(rgm_k,:,i));
            yi = squeeze(y(rgm_k,:,i));

            Sig_inv(:,:,i) = inv_v0 + xi' * xi;
            Sig_i = squeeze(Sig_inv(:,:,i))\speye(2);
            % rho_ik(:,i) = Sig_i * (inv_v0 * mu0' + xi'* yi);
            % b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i) + mu0sig0mu0); 
            rho_ik(:,i) = Sig_i * xi' * yi;
            b_ik(i) = 1/2 * (2 * b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i)); % replace mu0sig0mu0

            % draw sig2 - portfolio i
            sig2 = 1/gamrnd(a_k, 1/b_ik(i));

            % draw beta - portfolio i
            beta = mvnrnd(rho_ik(:,i), sig2 * Sig_i, 1);

            % store temp Sig_ik, a_k, b_ik
            temp_Sig(:,:,i,irgm) = Sig_i;
            temp_b(i,irgm) = b_ik(i);% bik is different in every portfolio

            % store beta and sig2
            temp_sig2(i,irgm) = sig2;
            temp_beta(i,:,irgm) = beta;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%  3.Change the number of break
    %%%  3.1 Birth move with 0.5 probability
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if num_regime == 1 % num_regime stays same/ K = 0
        p_birth = 1;
    elseif num_regime == T % num_regime max / K = T-1
        p_birth = 0;
    else
        p_birth = 0.5;
    end

    if rand() < p_birth
        stau = randi(T);
        % first, determine whether the new tau is similar to the current point
        if sum(tau==stau) == 0 % Same as the current break point-nothing happens
            regime_c = sum(tau<stau); % the current regime that will be separated

            l_kc = temp_l(regime_c); % current length of regime c
            Sig_kc = temp_Sig(:,:,:,regime_c);
            a_kc = temp_a(regime_c);
            b_kc = temp_b(:,regime_c);

            % create new temp srgm, using the previous one to save memory
            srgm_k = (tau(regime_c)+1):stau;
            srgm_k1 = (stau+1):tau(regime_c+1);

            sl_k = stau - tau(regime_c);
            sl_k1 = tau(regime_c+1) - stau;
            
            stemp_l = [sl_k,sl_k1];
            % get stemp_a
            stemp_a = [(a+sl_k/2),(a+sl_k1/2)];

            % new regime among the new break poiints
            for si = 1:2
                if si == 1
                    rgm_k = srgm_k;
                    a_k = stemp_a(1);
                else
                    rgm_k = srgm_k1;
                    a_k = stemp_a(2);
                end

                for i = 1:N
                    % new block
                    xi = x(rgm_k,:,i);
                    yi = y(rgm_k,:,i);

                    Sig_inv(:,:,i) = inv_v0 + xi' * xi;
                    Sig_i = Sig_inv(:,:,i)\speye(2);
                    % rho_ik(:,i) = Sig_i * (inv_v0 * mu0' + xi'* yi);
                    % b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'
                    % *Sig_inv(:,:,i)*rho_ik(:,i) + mu0sig0mu0); 
                    rho_ik(:,i) = Sig_i * xi'* yi;
                    b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i)); % replace mu0sig0mu0

                    % draw sig2 - portfolio i
                    sig2 = 1/gamrnd(a_k, 1/b_ik(i));

                    % draw beta - portfolio i
                    beta = mvnrnd(rho_ik(:,i),sig2 * Sig_i);

                    % store start-temp Sig_ik b_ik before and
                    stemp_Sig(:,:,i,si) = Sig_i;
                    stemp_b(i,si) = b_ik(i);% bik is different in every regime

                    % store beta and sig2
                    stemp_sig2(i,si) = sig2;
                    stemp_beta(i,:,si) = beta;
                end
            end

            % calculate the new acceptability probability
            alpha_birth = d^c * factorial(l_kc)*gamma(sl_k1+c)*gamma(sl_k+c)*(T-K)*b^(a*N)...
                /gamma(c)/factorial(sl_k)/factorial(sl_k1)/gamma(l_kc + c)/(d+1)^c/(K+1)...
                /((gamma(a))^N);% * p_birth/(1-p_birth);

            % get the value used in the following steps - sXXX is the new
            % regime associated with new break
            sak = stemp_a(1);sak1 = stemp_a(2);
            sbik = stemp_b(:,1);sbik1 = stemp_b(:,2);
            sSigk = squeeze(stemp_Sig(:,:,:,1)); sSigk1 = squeeze(stemp_Sig(:,:,:,2));

            for i =1:N
                alpha_birth = alpha_birth * gamma(sak) * gamma(sak1) * b_kc(i)^a_kc * sqrt(det(sSigk(:,:,i)))...
                    * sqrt(det(sSigk1(:,:,i)))/(sbik(i)^sak)/(sbik1(i)^sak1)/gamma(a_kc)/sqrt(det(v0))...
                    /sqrt(det(Sig_kc(:,:,i)));
            end

            if rand() < min(1,alpha_birth)
                tau = [tau(1:regime_c),stau,tau((regime_c + 1):end)] ;
                temp_l = [temp_l(1:(regime_c-1)),stemp_l,temp_l((regime_c + 1):end)];
                temp_a = [temp_a(1:(regime_c-1)),stemp_a,temp_a((regime_c + 1):end)];
                temp_b = cat(2,temp_b(:,1:(regime_c-1)),stemp_b,temp_b(:,(regime_c + 1):end));
                temp_Sig = cat(4,temp_Sig(:,:,:,1:(regime_c-1)),stemp_Sig,temp_Sig(:,:,:,(regime_c + 1):end));
                temp_sig2 = cat(2,temp_sig2(:,1:(regime_c-1)),stemp_sig2,temp_sig2(:,(regime_c + 1):end));
                temp_beta = cat(3,temp_beta(:,:,1:(regime_c-1)),stemp_beta,temp_beta(:,:,(regime_c + 1):end));
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%  Change the number of break
        %%%  3.2 Death move with 0.5 probability
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else
        stau = tau(randi([2,num_regime])); % random chose from the existing regime, excluding two end point 0 and T

        regime_c = sum(tau<stau); % the current regime that will be canceled before the break- k
        regime_c1 = regime_c+1; % the current regime that will be canceled - k + 1

        l_kc = temp_l(regime_c); l_kc1 = temp_l(regime_c1);
        a_kc = temp_a(regime_c); a_kc1 = temp_a(regime_c1);
        b_kc = temp_b(:,regime_c); b_kc1 = temp_b(:,regime_c1);
        Sig_kc = squeeze(temp_Sig(:,:,:,regime_c)); Sig_kc1 = squeeze(temp_Sig(:,:,:,regime_c1));

        %create new temp srgm, using the previous one to save memory
        rgm_k = (tau(regime_c)+1):tau(regime_c+2);
        sl_k = l_kc + l_kc1;

        % use the previous matrix to save space
        % ONLY USE FIRST DIMENSION
        si = 1;

        %get stemp_a
        stemp_a(si) = a+sl_k/2;
        a_k = stemp_a(si);

        % new regime among the new break poiints
        for i = 1:N
            %new block
            xi = x(rgm_k,:,i);
            yi = y(rgm_k,:,i);

            Sig_inv(:,:,i) = inv_v0 + xi' * xi;
            Sig_i = Sig_inv(:,:,i)\speye(2);
            % rho_ik(:,i) = Sig_i * (inv_v0 * mu0' + xi'* yi);
            % b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i) + mu0sig0mu0); 
            rho_ik(:,i) = Sig_i * xi'* yi;
            b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i)); % replace mu0sig0mu0

            %draw sig2 - portfolio i
            sig2 = 1/gamrnd(a_k, 1/b_ik(i));

            %draw beta - portfolio i
            beta = mvnrnd(rho_ik(:,i),sig2 * Sig_i);

            %store start-temp Sig_ik b_ik before and
            stemp_Sig(:,:,i,si) = Sig_i;
            stemp_b(i,si) = b_ik(i);% bik is different in every regime

            %store beta and sig2
            stemp_sig2(i,si) = sig2;
            stemp_beta(i,:,si) = beta;
        end

        %get the value used in the following steps - sXXX is the new
        %regime associated with new break
        sak = stemp_a(si);
        sbik = stemp_b(:,si);
        sSigk = squeeze(stemp_Sig(:,:,:,si));
        ssig2 = stemp_sig2(:,si);
        sbeta = stemp_beta(:,:,si);

        %calculate the new acceptability probability
        alpha_death = factorial(l_kc) * factorial(l_kc1) * gamma(sl_k + c)...
            *(d+1)^c * gamma(c) * K * (gamma(a))^N/factorial(sl_k)/d^c/...
            gamma(l_kc + c)/gamma(l_kc1 + c)/(T-K+1)/(b^(a*N));%  *(1-p_birth)/p_birth;

        for i =1:N
            alpha_death = alpha_death * b_kc(i)^a_kc * gamma(sak) * sqrt(det(sSigk(:,:,i))) * sqrt(det(v0)) * b_kc1(i)^a_kc1...
                / sbik(i)^sak/gamma(a_kc)/sqrt(det(Sig_kc(:,:,i)))/sqrt(det(Sig_kc1(:,:,i)))/gamma(a_kc1);
        end

        %% change the saved value
        if rand() < min(1,alpha_death)
            tau = [tau(1:regime_c),tau((regime_c + 2):end)];
            temp_l = [temp_l(1:(regime_c-1)),sl_k,temp_l((regime_c + 2):end)];
            temp_a = [temp_a(1:(regime_c-1)),sak,temp_a((regime_c + 2):end)];
            temp_b = [temp_b(:,1:(regime_c-1)),sbik,temp_b(:,(regime_c + 2):end)];
            temp_Sig = cat(4,temp_Sig(:,:,:,1:(regime_c-1)),sSigk,temp_Sig(:,:,:,(regime_c + 2):end));
            temp_sig2 = cat(2,temp_sig2(:,1:(regime_c-1)),ssig2,temp_sig2(:,(regime_c + 2):end));
            temp_beta = cat(3,temp_beta(:,:,1:(regime_c-1)),sbeta,temp_beta(:,:,(regime_c + 2):end));
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%  2.Perturb the current tau, holding the number of break constant
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % tau changed - recalculate the parameter
    num_regime = length(tau)-1;
    K = num_regime - 1;
    % Num of points being perturbed: num_regime - 1 (except two end points)
    if num_regime > 1 % only 1 regime, do not perturb
        temp_alpha_pert = zeros(1,K); % create vector to store alpha_pert (in current loop)
        
        itau = randi(K,1);

        %for itau = 1:K % pertube the change point between 0 and T
            taui = tau(itau+1); % old tau
            l_k = temp_l(itau);
            l_k1 = temp_l(itau+1);

            % s = min([l_k-1,l_k1-1,s0]); % make sure the new break point is between current break and the next break point

            %%  draw new tau
            s_new = randi([-s0,s0]);

            if (s_new ~= 0) && (((s_new>0)&&(s_new<l_k1))||((s_new<0)&&((-s_new)<l_k))) % ensure between two break
                tau_new = taui + s_new;
                %%  parameter after pertub
                srgm_k = (tau(itau)+1):tau_new;
                srgm_k1 = (tau_new +1):tau(itau+2);
                sl_k = tau_new - tau(itau);
                sl_k1 = tau(itau+2) - tau_new;
                stemp_a = [(a+sl_k/2),(a+sl_k1/2)];
                % calculate new sig, a, b and draw new beta, sig
                for si = 1:2
                    if si == 1
                        rgm_k = srgm_k;
                        a_k = stemp_a(1);
                    else
                        rgm_k = srgm_k1;
                        a_k = stemp_a(2);
                    end

                    for i = 1:N
                        xi = x(rgm_k,:,i);
                        yi = y(rgm_k,:,i);

                        Sig_inv(:,:,i) = inv_v0 + xi' * xi;
                        Sig_i = Sig_inv(:,:,i)\speye(2);
                        % rho_ik(:,i) = Sig_i * (inv_v0 * mu0' + xi'* yi);
                        % b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i) + mu0sig0mu0);
                        rho_ik(:,i) = Sig_i * xi'* yi;
                        b_ik(i) = 1/2 * (2*b + yi' * yi - rho_ik(:,i)'*Sig_inv(:,:,i)*rho_ik(:,i)); % replace mu0sig0mu0

                        % draw sig2 - portfolio i
                        sig2 = 1/gamrnd(a_k,1/b_ik(i));

                        % draw beta - portfolio i
                        beta = mvnrnd(rho_ik(:,i)',sig2 * Sig_i);

                        % store start-temp Sig_ik b_ik before and
                        stemp_Sig(:,:,i,si) = Sig_i;
                        stemp_b(i,si) = b_ik(i);% bik is different in every regime

                        % store beta and sig2
                        stemp_sig2(i,si) = sig2;
                        stemp_beta(i,:,si) = beta;
                    end
                end

                l_k = temp_l(itau);l_k1 = temp_l(itau+1); % restore old value

                ak = temp_a(itau);ak1 = temp_a(itau+1); % change ak
                sak = stemp_a(1);sak1 = stemp_a(2);

                bik = temp_b(:,itau);bik1 = temp_b(:,itau+1); % change bk
                sbik = stemp_b(:,1);sbik1 = stemp_b(:,2);

                Sigk = squeeze(temp_Sig(:,:,:,itau)); Sigk1 = squeeze(temp_Sig(:,:,:,itau+1));
                sSigk = squeeze(stemp_Sig(:,:,:,1)); sSigk1 = squeeze(stemp_Sig(:,:,:,2));

                %% start the calculation of alpha_pert
                alpha_pert = factorial(l_k)*factorial(l_k1)*gamma(sl_k+c)*gamma(sl_k1+c)...
                    /factorial(sl_k)/factorial(sl_k1)/gamma(l_k+c)/gamma(l_k1+c);

                for i = 1:N
                    alpha_pert = alpha_pert * sqrt(det(sSigk(:,:,i))) * sqrt(det(sSigk1(:,:,i)))...
                        * bik(i)^ak * bik1(i)^ak1/sqrt(det(Sigk(:,:,i)))/sqrt(det(Sigk1(:,:,i)))...
                        /sbik(i)^sak /sbik1(i)^sak1*gamma(sak)*gamma(sak1)/gamma(ak)/gamma(ak1);
                end

                temp_alpha_pert(itau) = min(1,alpha_pert); % acceptability probability

                if rand() <= temp_alpha_pert(itau) % replace old tau, otherwise, stays the same
                    tau(itau+1) = tau_new;
                    temp_a(itau:itau+1) = stemp_a;
                    temp_b(:,itau:itau+1) = stemp_b;
                    temp_Sig(:,:,:,itau:itau+1) = stemp_Sig;
                    temp_sig2(:,itau:itau+1) = stemp_sig2;
                    temp_beta(:,:,itau:itau+1) = stemp_beta;
                    temp_l(itau) = sl_k; temp_l(itau+1) = sl_k1;
                end
            end
        %end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%  4. Store Value - Final step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isim > burnin
        isave = isim - burnin;
        if mod(isave,thining)==0
            % store parameters
            isave = isave/thining;
            num_regime = length(tau)-1; % breakpoint + 2 end - 1  % number of regime or number of break point
            K = num_regime; % number of break
            for itau = 1:K
                rgm = (tau(itau)+1):tau(itau+1);
                lk = tau(itau+1) - tau(itau);
                store_sig2(:,rgm,isave) = repmat(temp_sig2(:,itau),1,lk);
                store_beta(:,:,rgm,isave) = repmat(temp_beta(:,:,itau),1,1,lk);
                store_tau(tau(itau+1),isave) = 1;
                store_alpha_pert(isave) = mean(temp_alpha_pert); % create vector to store alpha_pert (in current loop)
            end
        end
    end
    disp([num2str(isim),'th simulation finished']);
end

beta_ave = mean(store_beta,4);
beta_ave = squeeze(beta_ave(:,2,:));

hold on 
for i = 1:N
    plot(1:T, beta_ave(i,:));
    ylim([0 1]);
end
hold off


beta_error = beta_ave - beta_true;

hold on 
for i = 1:N
    plot(1:T, beta_error(i,:));
    ylim([0 1]);
end
hold off

sig2_ave = mean(store_sig2,3);
hold on 
for i = 1:N
    plot(1:T, sig2_ave(i,:));
    ylim([0 1]);
end
hold off

sig2_error = sig2_ave - sig2_true;
hold on 
for i = 1:N
    plot(1:T, sig2_error(i,:));
    ylim([0 1]);
end
hold off

% tau_ave = mean(sum(store_tau,1));

stem(1:T,mean(store_tau,2)*100,'LineStyle','-',"LineWidth",2,...
      'Marker',".");

% ylim([0 1])
mean(store_alpha_pert)