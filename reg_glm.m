
function results = reg_glm(y,mdl,opts,varargin)
% Fit regularized linear binning models
%
% Uses glmfitpq function by Patrick Mineault
% https://www.mathworks.com/matlabcentral/fileexchange/31661-fit-glm-with-quadratic-penalty
%

%{
% make X and regularizer (Q) matrices
opts=[];
opts.family = 'normid';
opts.algo='lbfgs';

y=pairdata.dff(:,2,1);
y=min(pairdata.dff(:,2,1),max(lin(pairdata.dff(:,1,1))).*0.2);
%dat=([pairdata.xy_filt']);

%{[DATA(samples,dims)],'type',[Nbins ..],[range ..],[amp weights],[smooth_weights ..],[circular ..]}

mdl = {[],'bias',[],[],[0],[],[],...
    [pairdata.xy_filt'],'tile_laplace',[15 15],[[-0.4 0.4];[-0.4 0.4]],[2],[4 4],[0 0],...
    [pairdata.angle_filt'],'tile_laplace',[25],[0 2*pi],[1],[15],[1]...
    [min(pairdata.speed',6) pairdata.turn_speed_clipped'],'tile_laplace',[10 10],[[0 6];[-1 1]],[1],[5 5],[0 0],...
    [d_obj{1}' phi_obj{1}'], 'tile_laplace',[15 15],[[0 1];[-pi pi]],[0],[10 10],[0 1],...
    [d_obj{2}' phi_obj{2}'], 'Otile_laplace',[1 15],[[0 1];[-pi pi]],[1],[5 5],[0 0],...
    [d_obj{3}' phi_obj{3}'], 'Otile_laplace',[15 15],[[0 1];[-pi pi]],[1],[5 5],[0 0],...
    };
%}

%mdl = {'bias',[],[],[0],[],[], 'tile_laplace',[12 12],[[-0.4 0.4];[-0.4 0.4]],[2],[2 2],[0 0],'tile_laplace',[25],[0 2*pi],[1],[1],[1]} ;
%mdl = {'bias',[],[],[0],[],[], 'tile_laplace',[10 10 16],[[-0.4 0.4];[-0.4 0.4];[0 2*pi]],[2],[1 1 2],[0 0 1],} ;


params_per_mdl=7;

N_independent_mdls=size(mdl,2)/params_per_mdl;

assert(mod(N_independent_mdls,1)==0,'model parameters must be multiple of 6: {''type'',[Nbins ..],[range ..],[amp weights],[smooth_weights ..],[circular ..]}');

% make regressor array
dat=[];
for b=1:N_independent_mdls
    ofs=(b-1)*params_per_mdl;
    dat=[dat mdl{ofs+1}];
end;
Ndat=size(dat,1);

assert(size(y,1)==Ndat,'regressors in mdl must have same N of samples as y');

if numel(varargin)==0 % fit/eval on whole dataset
    samples_fit=logical(ones(Ndat,1));
    samples_eval=logical(ones(Ndat,1));
else % fit/eval on subsets
    samples_fit=logical(varargin{1});
    samples_eval=logical(varargin{2});
end;
assert(all(size(samples_fit)==size(y)),'fit/eval selection must be logical of asme size as y');
assert(all(size(samples_eval)==size(y)),'fit/eval selection must be logical of asme size as y');

total_dims=0;
for b=1:N_independent_mdls
    total_dims =total_dims+ size(mdl{((b-1)*params_per_mdl)+3},2);
end;
assert(size(dat,2)==total_dims,'data dims must match model dims');


X=sparse([]);
Q=sparse([]);
param_end=0;
dim_block_offset=0; % offset by where we ingest data from
block_num = []; % for reshaping later
overlay_onto_prev_bins=[]; % to keep track of which model blocks are now empty because they were folded into previous ones

for b=1:N_independent_mdls
    ofs=(b-1)*params_per_mdl;
    type=mdl{ofs+2};
    
    
    t=type;
    type=strip(type,'left','O'); % prefix model with 'O' to overlay bin occupancy with previous model block. 
                                 % Can be used for models with multiple occupancy, for instance position of multiple landmarks
    if strcmp(t,type)
        overlay_onto_prev_bins(b)=0;
        
        Nbins=mdl{ofs+3}; % if overlaying just keep all parameters from prev.
        range=mdl{ofs+4};
        amp_weight=mdl{ofs+5};
        gauss_weights=mdl{ofs+6};
        circular=mdl{ofs+7};
        
    else
        overlay_onto_prev_bins(b)=1;
        amp_weight=amp_weight.*0;  % if overlaying, just use regularization parameters from first mdl block
        gauss_weights=gauss_weights.*0;
    end
    
    switch lower(type)
        case 'bias'
            param_end=param_end+1;
            X=[X, ones(Ndat,1)]; % add ones
            Q(param_end,param_end)=amp_weight; % add L2 weight reg.
            block_num(end+1)=b;
            
        case 'tile_laplace'
            Ndims = size(Nbins,2);
            Nparams=prod(Nbins);
            if overlay_onto_prev_bins(b) ==0 % for overlaying just do everything the same way but dont increment parameter counts
                param_start=param_end+1;  % find range in X/Q that we're dealing with
                param_end=param_end+Nparams;
            end;
            p_range=[param_start:param_end];
            
            if overlay_onto_prev_bins(b) ==0
                block_num(p_range)=b;
            end;
            % bins for X
            clear ll;
            for d=1:Ndims
                ll{d}=linspace(range(d,1),range(d,2),Nbins(d)+1);
            end;
            
            dim_boundaries = NaN(Nparams,Ndims,2); % for binning
            
            Q(param_end,param_end)=0; %expand Q
            
            % do L2 amp reg  in Q
            Q(p_range,p_range) = Q(p_range,p_range) + eye(Nparams).*amp_weight;
            
            %do laplace operator in Q
            for d=1:Ndims
                % populate Q
                nb_pad=[1 Nbins];
                stride= max(1,prod(nb_pad(1:d)));
                
                blksize=Nbins(d)*stride;
                Nblks=Nparams/blksize;
                
                block_template =  diag(ones(blksize,1).*-2,0) + diag(ones(blksize-stride,1),stride) + diag(ones(blksize-stride,1),-stride);
                if circular(d)
                    block_template =block_template  + diag(ones(stride,1),blksize-stride)+diag(ones(stride,1),-(blksize-stride));
                else
                    for i=1:stride
                        block_template(i,i)=-1;
                        block_template(end-(i-1),end-(i-1))=-1;
                    end;
                end
                
                for block=1:Nblks
                    p_range_block=[param_start:param_start+blksize-1]+(block-1)*blksize;
                    Q(p_range_block,p_range_block)=Q(p_range_block,p_range_block) + block_template.*gauss_weights(d);
                    
                    ll_lo=ll{d}(1:end-1);
                    ll_hi=ll{d}(2:end);
                    
                    % save bin boundaries for X
                    ii=[1:numel(p_range_block)];
                    dim_boundaries(p_range_block,d,1)=ll_lo(ceil(ii./stride));
                    dim_boundaries(p_range_block,d,2)=ll_hi(ceil(ii./stride));
                    
                end;
            end; % trough dims for laplace
            
            % pre-allocate X
            X(param_end,param_end)=0;
            
            for p=1:Nparams
                ii= ones(Ndat,1);
                for d=1:Ndims
                    ii = ii & dat(:,d+dim_block_offset) >= dim_boundaries(p_range_block(p),d,1)  & dat(:,d+dim_block_offset) <= dim_boundaries(p_range_block(p),d,2);
                end;
                
                X(:,p+(param_start-1))=X(:,p+(param_start-1))+ii; % add here instead of overwriting so that we can overlay occupancy
            end;
            
            dim_block_offset=dim_block_offset+Ndims;
        case 'history'
            % 2do: add 
            
        case 'poly'
            
    end;
end;


%
% now fit model
results = glmfitqp(circshift(y(samples_fit),0), X(find(samples_fit),:), Q, opts);

% reshape results
for b=1:N_independent_mdls
    if ~overlay_onto_prev_bins(b)
        dims=mdl{((b-1)*params_per_mdl)+3};
        if numel(dims)==0; dims=[1]; end;
        if numel(dims)==1; dims=[dims 1]; end;
        
        w_block = results.w(block_num==b);
        
        results.reshaped{b}=reshape(w_block,dims);
    else
        results.reshaped{b}=[];
    end;
end;

% make prediction
assert(   any(strcmp(opts.family,{'normid','binomlogit'}))  ,'dont have poisson implemented yet');

switch opts.family
    case 'normid'
        
        results.y_pred=X(samples_eval,:)*results.w;
        results.rmse=sqrt( mean((y(samples_eval)-results.y_pred).^2) )
        
    case 'binomlogit'
        
        results.y_pred =sigmoid(X(samples_eval,:)*results.w);          
        results.rmse=mean(results.y_pred>0.5==y(samples_eval));
        
        %{
        clf; hold on;
        plot(y(samples_eval));
        plot( results.y_pred)
        %}
        
end;
results.mdl=mdl;
results.N_independent_mdls=N_independent_mdls;

%% plot the results
if 0
    
    figure(3);
    clf; hold on;
    for b=1:results.N_independent_mdls
        
        Nnonsingleon= sum(size(results.reshaped{b})>1);
        
        subplot(3,3,b);
        switch Nnonsingleon
            case 0
                plot( results.reshaped{b});
                title(num2str(results.reshaped{b}))
            case 1
                plot( results.reshaped{b});
            case 2
                imagesc( results.reshaped{b});
                daspect([1 1 1])
            case 3
                subplot(3,3,[4 5 6]);
                I=[];
                for ldim=1:size(results.reshaped{b},3)
                    I=[I results.reshaped{b}(:,:,ldim)];
                end;
                imagesc( I);
                daspect([1 1 1])
        end;
        
    end;
    
    subplot(3,3,[7:9]); hold on;
    plot(y(samples_eval)-1.1);
    plot(sigmoid(X(samples_eval,:)*results.w))
    ylim([-1.2 1.1]);
    colormap viridis;
    
    
    
end;





