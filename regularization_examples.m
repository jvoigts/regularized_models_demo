x=[1:10,1:.01:2, 6:.2:7];
y=sin(x.*5);
y(106)=0;



ll=linspace(0,10,100); % bins

X=ones(size(x))';

for i=1:numel(ll)-1
   X=[X,(x>ll(i) & x<=ll(i+1))'];
end;

X=X(:,2:end); % better get rid of offset term

figure(1); clf; hold on;

subplot(331); hold on;
plot(x,y,'co');
b=regress(y',X);
plot(x,X*b,'ro');
plot(ll(2:end),b,'b')


b0=zeros(1,size(X,2));

subplot(332); hold on;
plot(x,y,'co');
% least sq again, but now lets just minimize the dumb way
b=fminunc(@(b) norm((b*X')-y)^2,b0);
plot(x,X*b','ro');
plot(ll(2:end),b,'b')

subplot(333); hold on;
plot(x,y,'co');
% now with L2 penalty on b
b=fminunc(@(b) norm((b*X')-y)^2 + 2*norm(b)^2,b0);
plot(x,X*b','o');
plot(ll(2:end),b,'b')

subplot(334); hold on;
plot(x,y,'co');
% and L2 penalty on paired differences
b=fminunc(@(b) norm((b*X')-y)^2 + .1*norm(diff(b))^2,b0);
plot(x,X*b','ro');
plot(ll(2:end),b,'b')


subplot(335); hold on;
plot(x,y,'co');
% and L2 penalty on paired differences, higher weight
b=fminunc(@(b) norm((b*X')-y)^2 + 5*norm(diff(b))^2,b0);
plot(x,X*b','ro');
plot(ll(2:end),b,'b')


subplot(336); hold on;
plot(x,y,'co');
% and L2 penalty on paired differences, 2nd order
b=fminunc(@(b) norm((b*X')-y)^2 + 5*norm(diff(diff(b)))^2,b0);
plot(x,X*b','ro');
plot(ll(2:end),b,'b')


subplot(337); hold on;
plot(x,y,'co');
% and L2 penalty on paired differences, plus weigth term
b=fminunc(@(b) norm((b*X')-y)^2 + 1*norm(diff(b))^2 + 2*norm(b)^2 ,b0);
plot(x,X*b','ro');
plot(ll(2:end),b,'b')