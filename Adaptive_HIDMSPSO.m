% -----------------------------------------------------------------------  %
% HIDMS-PSO Algorithm with an Adaptive Topological Structure               %
%
% Implemented by Fevzi Tugrul Varna - University of Sussex                 %
% -------------------------------------------------------------------------%

% Cite as: ----------------------------------------------------------------%
% F. T. Varna and P. Husbands, "HIDMS-PSO Algorithm with an Adaptive       %
% Topological Structure," 2021 IEEE Symposium Series on Computational      %
% Intelligence (SSCI), Orlando, FL, USA, 2021, pp. 1-8, doi:               %
% 10.1109/SSCI50451.2021.9660115.                                          %
% -----------------------------------------------------------------------  %
function [fmin] = Adaptive_HIDMSPSO(fhd,fId,n,d,range)
if rem(n,4)~=0, error("** Input Error: Swarm population must be divisible by 4 **"), end
rand('seed',sum(100*clock));
showProgress=true;
Fmax=10^4*d;      %maximum number function evaluations
Tmax=Fmax/n;      %maximum number of iterations
LB=range(1);
UB=range(2);
%% Parameters of Adaptive-HIDMS-PSO
w1 = 0.99 + (0.2-0.99)*(1./(1 + exp(-5*(2*(1:Tmax)/Tmax - 1)))); %nonlinear decrease inertia weight - Sigmoid function
c1=2.5-(1:Tmax)*2/Tmax;             %personal acceleration coefficient
c2=0.5+(1:Tmax)*2/Tmax;             %social acceleration coefficient          

UPn=4;                              %unit pop size (constant)
U_n=(n/2)/UPn;                      %number of units (constant)
U=reshape(randperm(n/2),U_n,UPn);   %units (U_n-by-UPn matrix)
unit_T=zeros(1,U_n);
[master,s1,s2,s3] = feval(@(x) x{:}, num2cell([1,2,3,4])); %unit members' codes
formation=[1 1 1 1; 1 1 4 3; 1 1 2 2; 1 1 1 3; 1 1 randi([1 4]) randi([1 4])];

%velocity clamp
MaxV = 0.15*(UB-LB);
MinV = -MaxV;

%% Initialisation
V=zeros(n,d);           %initial velocities
X=unifrnd(LB,UB,[n,d]); %initial positions
S=zeros(1,n);           %subordinate status, 0 particle has no subordinate, 1 has.
PX=X;                   %initial pbest positions
F=feval(fhd,X',fId);    %function evaluation
PF=F;                   %initial pbest cost
GX=[];                  %gbest solution vector
GF=inf;                 %gbest cost

%update gbest
for i=1:n 
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end 
end

M_phase=true;           %master-dominated units phase
S_phase=false;          %slave-dominated units phase
gamma=100;              %number of iterations for each phase e.g. master-dominated or slave-dominated

%% Main Loop of PSO
for t=1:Tmax

    %switch between the master-dominated and slave-dominated units every 100 iterations
    if mod(t,gamma)==0
        if M_phase==true
            M_phase=false;
            S_phase=true;
            formation=[1 2 3 4; 1 2 2 3; 2 2 3 4; 1 3 4 4; 2 3 randi([2 4]) randi([2 4])];
        elseif S_phase==true
            S_phase=false;
            M_phase=true;
            formation=[1 1 1 1; 1 1 4 3; 1 1 2 2; 1 1 1 3; 1 1 randi([1 4]) randi([1 4])];
        end
    end

    for i=1:n
        if F(i) >= mean(F)
            w = w1(t) + 0.15;
            if w>0.99,  w=0.99; end
        else
            w = w1(t) - 0.15;
            if w<0.20,  w=0.20; end
        end

        if t<=Tmax*0.9
            if ~isempty(find(U==i))                     %if agent is part of a unit
                [uId,pId]=find(U==i);                   %get unit id and particle id
                unit_formation=formation(uId,:);        %get formation
                topology=GT(unit_formation);            %get topology
                unit_T(uId)=topology;                   %update topology

                %if ith particle is subordinate, it may switch its role
                if S(i)==1
                    if randi([0 1])==0
                        roles=getRoles(1);                          %get random role for the subordinate
                        formation(uId,pId)= roles;                  %new role of the subordinate particle
                        topology=GT(formation(uId,:));              %get the new topology (incase it changed)
                        unit_T(uId)=topology;                       %update topology of the unit
                        S(U(uId,:))=0;                              %reset subordinates after new topology
                        if unit_T(uId)==2 || unit_T(uId)==3 || unit_T(uId)==4 || unit_T(uId)==7     %check if any unit has subordinate member
                            subIds=FS(unit_T(uId),formation(uId,:));
                            S(U(uId,subIds))=1;                     %update subordinate status for the new formation
                        end
                    end
                end

                if topology==1
                    if pId==master                                              %if particle is master, always use outward-oriented movement
                        behaviour=randi([1 3]);
                        if behaviour==1                                         %master moves towards the avg position of another unit
                            x_unit_avg=mean(X(U(GRU(U_n,uId),:),:));
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(x_unit_avg - X(i,:));
                        elseif behaviour==2                                     %master moves towards the master of a randomly selected unit
                            x_unit_m=X(U(GRU(U_n,uId),master),:);
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(x_unit_m - X(i,:));
                        elseif behaviour==3
                            x_unit_m=X(U(GRU(U_n,uId),master),:);               %master of a randomly selected unit
                            x_avg=mean(X(U(uId,s1:s3),:));                      %avg position of own slave particles
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(x_avg - X(i,:)) + c2(t)*rand([1 d]).*(x_unit_m - X(i,:));
                        end
                    else                                                        %for slave particles randomly select a behaviour
                        behaviour=randi([0 1]);
                        if behaviour==0                                         
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(uId,find(formation(uId,:)==master)),:) - X(i,:)); %inward-oriented - position of the master
                        else
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(GRU(U_n,uId),pId),:) - X(i,:)); %outward-oriented - slave moves towards another slave of the same type in random unit
                        end
                    end
                elseif topology==2 || topology==3
                    if pId==master                                              %if particle is master, always use outward-oriented movement
                        behaviour=randi([1 3]);
                        if behaviour==1                                         %master moves towards avg position of another unit
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(X(U(GRU(U_n,uId)),:)) - X(i,:));
                        elseif behaviour==2                                     %master moves towards the master of a random unit
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(GRU(U_n,uId),master),:) - X(i,:));
                        elseif behaviour==3
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(mean(X(U(uId,s1:s3),:)) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(GRU(U_n,uId),master),:) - X(i,:));
                        end
                    else    %for slave particles randomly select a behaviour
                        if S(i)==1                                              %if slave is subordinate, move towards superior
                            %find the superior particle
                            fcopy=formation(uId,:);
                            roleVal=fcopy(pId);
                            fcopy(pId)=-1;
                            [~,idx]=find(fcopy==roleVal);
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(uId,idx),:) - X(i,:));
                        else                                                    %if slave particle has no subordinate
                            behaviour=randi([0 1]);
                            if behaviour==0                                     %inward-oriented movement
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(uId,find(formation(uId,:)==master)),:) - X(i,:)); %move towards the position of the master
                            else                                                %outward-oriented movement
                                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(GRU(U_n,uId),pId),:) - X(i,:)); %slave moves towards another slave of the same type in random unit
                            end
                        end
                    end
                elseif topology==6 || topology==7 || topology==8
                    if pId==master                                              %if particle is master, use outward-oriented movement
                        behaviour=randi([1 3]);
                        if behaviour==1                                         %master particle moves towards the most dissimilar slave particle
                            x_dis=X(U(uId,GDS(X(U(uId,pId),:),X(find(U(2,:)>master),:))),:);    %position of the most dissimilar slave
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(x_dis - X(i,:));
                        elseif behaviour==2                                     %master particle moves towards the slave with lowest cost
                            slaves=U(uId,find(formation(uId,:)>master));        %only get slave particles in the same unit
                            [~,bestId]=min(F(slaves));                          %find the slave particle with the lowest cost in the unit
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(slaves(bestId),:) - X(i,:)); %move towards the position of the slave with lowest cost in the unit
                        elseif behaviour==3                                     %master particle moves towards avg slave positions
                            slaves=U(uId,find(formation(uId,:)>master));        %only get slave particles in the unit
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(X(slaves,:)) - X(i,:)); %move towards the avg position of slaves
                        end
                    else    %slave particles randomly select a behaviour
                        behaviour=randi([0 1]);
                        if behaviour==0                                         %inward-oriented movement
                            x_m=X(U(uId,find(formation(uId,:)==master)),:);     %position of the master
                            [mN,~]=size(x_m);                                   %number of masters unit has
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(x_m(randi([1 mN]),:) - X(i,:));
                        else                                                    %outward-oriented movement
                            V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(U(GRU(U_n,uId),pId),:) - X(i,:));  %slave particle moves towards another slave of the same type in a randomly selected unit
                        end
                    end
                elseif topology==4 || topology==5
                    members=U(uId,:);
                    members(pId)=[];
                    V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(X(members,:)) - X(i,:));
                end
            else                                                                %non unit member velocity update
                V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            end
        end

        if t > Tmax*0.9, V(i,:) = w*V(i,:) + c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:)); end

        V(i,:)=max(V(i,:),MinV); V(i,:) = min(V(i,:),MaxV);                     %velocity clamp
        X(i,:)=X(i,:) + V(i,:);                                                 %update position
        if t <= Tmax*0.9, X(i,:) = MUT(X(i,:),0.1,t,Tmax,[LB UB]); end          %use mutation operator in the final phase of the search
        X(i,:)=max(X(i,:), LB); X(i,:) = min(X(i,:), UB);                       %apply lower and upper bound limits
    end

    F=feval(fhd,X',fId);                                                        %fitness evaluation

    for j=1:n
        if F(j)<PF(j), PF(j)=F(j); PX(j,:)=X(j,:); end  %update pbests
        if PF(j)<GF, GF=PF(j); GX=PX(j,:); end          %update gbest
    end

    if showProgress, disp(['Iteration '   num2str(t)  ': best cost = '  num2str(GF)]); end
end

fmin = GF;
end

%% generates and returns random roles
%% args: N: number of roles to generate
function [roles] = getRoles(N)
roles=randi([0 1],1,N);
for i=1:length(roles)
    if roles(i)==0, roles(i)=randi([2 4]); end
end
end

%% returns the most dissimilar slave particle to master
%% args: m: master particle position, s: position of the slave particles
function [z] = GDS(m,s)
L=size(s);
score=zeros(1,L(1));

for i=1:L(1)
    score(i)=immse(m,s(i,:));
end

[~,z]=max(score);
end

%% returns a random unit
%% args: U_n: number of units, uId: id of the current unit
function [z] = GRU(U_n,uId)
rndU=randperm(U_n);
self=find(rndU==uId);
rndU(self)=[];
z=rndU(1);
end

%% returns the topology of the given unit
function topId = GT(unit)
unit=sort(unit); 
s=[2 3 4];
m=1;

if sum(unit==[1 2 3 4])==4, topId=1;
elseif unit(1)==m && length(unique(unit(2:4)))==2 && sum(unit(2:4)~=1)==3, topId=2;
elseif isempty(find(unit==m))==1, topId=4;
else
    if sum(ismember(unit,[m m m m]))==4, topId=5;   %topology with all masters
    elseif sum(unit(1:2)==[m m]) && length(unique(unit(3:4)))==2 && unit(3)~=1, topId=6;
    elseif sum(unit(1:2)==[m m]) && length(unique(unit(3:4)))==1, topId=7;
    elseif sum(ismember(unit(1:3),[m m m]))==3 && ismember(unit(4),s)==1, topId=8;
    end
end
end

%% returns the subordinate particle given the unit topology and unit formation
%% args: T: topology of the unit, UF: unit formation
%% returns a vector of 0's and 1' where 1's refer to id of the subordinate member
function [z] = FS(T,UF)
%z - returns a vector of 0's and 1', index of 1's refer to id of the subordinate member
z=ones(1,4); %0 or 1 for each unit member indicating weather member is sub or not

if T==2 || T==3 || T==4 || T==7
    [~,idx]=unique(UF); %find duplicates
    z(idx)=0;           %change non-subordinates to 0
end

z=find(z==1); %find the id of the subordinates

end

%% non-uniform mutation
function [y] = MUT(x,p,t,Tmax,range)
b=5;                %system parameter - 2~5
[m,n]=size(x);
y = x;
for i=1:m
    for j=1:n
        if rand<p
            D = diag(rand(1,n));
            if round(rand)==0
                y(i,j) = x(i,j)+D(j,j)*(range(2)-x(i,j))*(1-t/Tmax)^b;
            else
                y(i,j) = x(i,j)-D(j,j)*(x(i,j) - range(1))*(1-t/Tmax)^b;
            end
        end
    end
end
end