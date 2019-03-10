function [ll g Hd r lleach] = evalGlmLikelihood(y,X,w,b,family,familyextra,weights)
    %[ll g] = evalGlmLikelihood(y,X,w,b,family,familyextra,weights)
    %Evaluates the negative log-likelihood of the data assuming a GLM
    %with the specified properties
    
    if nargin < 7
        weights = 1;
    end
    
    r = X*w + b;
    switch family
        case 'normid'
            v = r;
            gain = 1/familyextra^2;
            lleach = .5/familyextra^2*(y-v).^2;
        case 'binomlogit'
            if familyextra == 1
                %Can skip a few computations in this case
                gain = 1;
                v = 1./(1+exp(-r));
                lleach = zeros(size(y));
                lleach(y==1) = -log(v(y==1)+eps);
                lleach(y==0) = -log(1-v(y==0)+eps);
            else
                y = y/familyextra;
                v = 1./(1+exp(-r));
                gain = familyextra;

                lleach = -familyextra*(y.*log(v+eps) + ...
                                   (1-y).*log(1-v+eps));
            end
        case 'poissexp'
            v = exp(r);
            gain = 1;
            lleach = -y.*r + v;
        otherwise 
            error('Unsupported family');
    end
    
    lleach = lleach.*weights;
    ll = sum(lleach);
    
    if nargout > 1
        %These are all canonical links
        r = weights.*(v-y);
        g = gain*(X'*r);
    end
    if nargout > 2
        %Compute the diagonal term that occurs in the Hessian (canonical
        %links only)
        switch family
            case 'normid'
                Hd = 1/familyextra^2*ones(size(X,1),1).*weights;
            case 'binomlogit'
                Hd = familyextra*v.*(1-v).*weights;
            case 'poissexp'
                Hd = v.*weights;
        end
    end
end