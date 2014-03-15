function K = kernel(X1, X2,kernelpar,kertype)
% SVMKERNEL - Compute Support Vector Machine kernel function
% 
%   K = SVMKERNEL(NET, X1, X2)
%   The Support Vector Machine structure NET must contain 2 field
%   NET.kernel and NET.kernelpar, selecting the kernel function and its
%   parameters. X1 and X2 contain one example per row. If X1 is of size
%   [M, NET.nin] and X2 is of size [N, NET.nin], K will be a matrix
%   [M, N]. K(i,j) is the result of the kernelfunction for inputs X1(i,:)
%   and X2(j,:).
%   Currently the only valid kernel functions are
%   NET.kernel = 'linear' 
%       inner product
%   NET.kernel = 'poly'
%       (1+inner product)^NET.kernelpar(1)
%   NET.kernel = 'rbf'
%       radial basis function, common length scale for all inputs is
%       NET.kernelpar(1), scaled with the number of inputs NET.nin
%       K = exp(-sum((X1i-X2i)^2)/(NET.kernelpar(1)*NET.nin))
%   NET.kernel = 'rbffull'
%       radial basis function, different length scale for each input.
%       If NET.kernelpar is a vector of length NET.nin
%       K = exp(-sum((X1i-X2i)^2*NET.kernelpar(i))/NET.nin)
%       If NET.kernelpar is a vector of length NET.nin+1
%       K = exp(NET.kernelpar(end)-sum((X1i-X2i)^2*NET.kernelpar(i))/NET.nin)
%       If NET.kernelpar is a matrix of size [NET.nin, NET.nin]
%       K = exp(-(X1-X2)*NET.kernelpar*(X1-X2)'/NET.nin)
%
%   See also
%   SVM, SVMTRAIN, SVMFWD
%


% 
% Copyright (c) Anton Schwaighofer (2001)
% $Revision: 1.3 $ $Date: 2001/06/18 15:21:55 $
% mailto:anton.schwaighofer@gmx.net
% 
% This program is released unter the GNU General Public License.
% 
[N1, d] = size(X1);
[N2, d] = size(X2);
switch kertype
  case 'linear'
    K = X1*X2';
    case 'linear1'
     K1=X1*X2';
     K2=max(diag(X1*X1'));
     K=K1+diag(sqrt(K2*K2-diag(X1*X1')-diag(X2*X2')));
  case 'poly'
    K = (1+X1*X2').^kernelpar;
  case 'rbf1'
    X11 = X1(:,1:d-1);
    X22 = X2(:,1:d-1);
    dist2 = repmat(sum((X11.^2)', 1), [N2 1])' + ...
           repmat(sum((X22.^2)',1), [N1 1]) - ...
           2*X11*(X22');
    K1 = X1(:,d)*X2(:,d)';
    K = exp(-dist2/((d-1)*kernelpar))+K1;%核函数分为径向基和线性两部分
   % K = exp(-dist2/(net.nin*net.kernelpar(1)));
    %dist2 = repmat(sum((X1.^2)', 1), [N2 1])' + ...
     %       repmat(sum((X2.^2)',1), [N1 1]) - ...
      %      2*X1*(X2');
   % K = exp(-dist2/(net.nin*net.kernelpar(1)));
   case 'rbf'
    dist2 = repmat(sum((X1.^2)', 1), [N2 1])' + ...
            repmat(sum((X2.^2)',1), [N1 1]) - ...
           2*X1*(X2');
    K = exp(-dist2/(d*kernelpar));
case 'rbfsp'
    X11 = X1(:,1:d-2);
    X22 = X2(:,1:d-2);
   
    dist2 = repmat(sum((X11.^2)', 1), [N2 1])' + ...
           repmat(sum((X22.^2)',1), [N1 1]) - ...
           2*X11*(X22');
    dist11 = repmat(sum((X11.^2)', 1), [N1 1])' + ...
           repmat(sum((X11.^2)',1), [N1 1]) - ...
           2*X11*(X11');
    dist22 = repmat(sum((X22.^2)', 1), [N2 1])' + ...
           repmat(sum((X22.^2)',1), [N2 1]) - ...
           2*X22*(X22');
    K11= exp(-dist11/((d-1)*kernelpar));  
    K22= exp(-dist22/((d-1)*kernelpar));
    mK11=max(K11);
    mK22=max(K22);
%     save mK22  mK22;
%     save mK11  mK11;
%     save K11  K11;
%     save K22  K22;
%     save X1 X1;
%     save X2 X2;
%     save dist2 dist2;
    
    K1 = X1(:,d)*X2(:,d)';;
    K2 = exp(-dist2/((d-1)*kernelpar));

    K3=sqrt((mK11'*mK22).^2-(K11*mK22')*((K11*mK22')')-(K22*mK11')*((K22*mK11')')+1);
    K=K1+K2+K3; 
   
  case 'rbffull'
    bias = 0;
   % if any(all(repmat(size(kernelpar), [4 1]) == [d 1; 1 d; d+1 1; 1 d+1], 2), 1),
      weights = diag(kernelpar(1:d));
      bias=0;
%       if length(kernelpar)>d,
%         bias = kernelpar(end);
%       end
%     elseif all(size(kernelpar)==[d d]),
%       weights = kernelpar;
%     else
%       error('Size of NET.kernelpar does not match the chosen kernel ''rbffull''');
%     end
    dist2 = (X1.^2)*weights*ones([d N2]) + ...
            ones([N1 d])*weights*(X2.^2)' - ...
            2*X1*weights*(X2');
    K = exp(bias-dist2/d-1);
    %end
%   otherwise
%     error('Unknown kernel function');
end

%K = double(K);

% Convert to full matrix if inputs are sparse


