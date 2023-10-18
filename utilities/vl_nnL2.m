function Y = vl_nnL2(X,c,dzdy,varargin)

% --------------------------------------------------------------------
% pixel-level L2 loss
% --------------------------------------------------------------------
%% Reshape the X to match the dimension of the label
Num_mat = size(c,3); 
X_rep = repmat(X,[1,1,Num_mat,1]);
dif = X_rep-c; t = dif.^2/2;
dist = sum(sum(t));
[m_val,m_ind] = min(dist);
index_i = m_ind(:);
c_opt = zeros(size(X));
index = sub2ind([size(c,3),size(c,4)],index_i,1:length(index_i));
c_opt(:,:,1,:) = c(:,:,index); %c_opt is the best matched labels

if nargin <= 2 || isempty(dzdy)
    Y = sum(m_val)/size(X,4); % reconstruction error per sample;
else
    Y = bsxfun(@minus,X,c_opt).*dzdy;
end

