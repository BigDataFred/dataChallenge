%function dataMining(p2d)
%%
%if nargin ==0
    %p2d = '/home/adf/rouxf/Downloads/data-lover-challenge/csvDat/';
    p2d = '/Users/froux/Documents/dataChallengeXabet/dataChallenge/csvDat/';
%end;

%%
[files] = dir([p2d,'*.csv'])

%%
bigD = [];
for it = 1:length(files)
        
    fid = fopen([p2d,files(it).name],'r');
    dat = textscan(fid,'%s');
    dat = dat{:};
    
    ix = regexp(dat{1},',');
    hdr = cell(1,length(ix));
    for jt = 1:length(ix)+1
        
        if (jt > 1) && (jt < length(ix)+1)
            x = dat{1}(ix(jt-1)+1:ix(jt)-1);
        elseif jt ==1
            x = dat{1}(1:ix(jt)-1);
        else
             x = dat{1}(ix(jt-1)+1:end);
        end;
        hdr(jt) = {x};
    end;
    
    X = cell(length(dat)-1,length(hdr));
    c = 0;
    for kt = 2:length(dat)
        c = c+1;
        ix = regexp(dat{kt},',');
        M = cell(1,length(ix));
        for jt = 1:length(ix)+1
            
            if (jt > 1) && (jt < length(ix)+1)
                x = dat{kt}(ix(jt-1)+1:ix(jt)-1);
            elseif jt ==1
                x = dat{kt}(1:ix(jt)-1);
            else
                x = dat{kt}(ix(jt-1)+1:end);
            end;
            M(jt) = {x};            
        end;
        X(c,:) = M;
    end;
    bigD = [bigD;X];
end;
clear dat;

%%
M = str2double(bigD(:,1:end-1));
tp = bigD(:,end);
clear bigD;

%%
tpID = unique(tp);
lab = zeros(length(tp),1);
for it = 1:length(tpID)
    lab(find(strcmp(tp,tpID(it)))) = it;
end;
M(:,end+1) = lab;
labId = unique(lab);

seg = [1;find([0;diff(M(:,2))]==1);size(M,1)];

%%
C = {'r','y','g','c','b'};

dId = unique(M(:,1));
n1 = [];
for it = 1:length(dId)
    dIx = find(M(:,1)==dId(it));
    [n1(it,:),x1] = hist(lab(dIx),[1:length(tpID)]);
end;
[n2,x2] = hist(M(:,2),sort(unique(M(:,2))));
figure;
subplot(421);
imagesc(M);
set(gca,'XTick',1:length(hdr));
set(gca,'xTickLabel',hdr);
ylabel('data samples');
set(gca,'XTickLabelRotation',45)
subplot(422);
hold on;
%[ax,h1,h2] = plotyy(1:size(M,1),M(:,2),1:size(M,1),[0;diff(M(:,2))]);
for it = 1:length(seg)-1
    h = area([seg(it) seg(it+1)-1],[max(M(:,2)) max(M(:,2))],min(M(:,2)));
    if mod(it,2) == 0
        set(h,'FaceColor',[1 1 1],'EdgeColor',[1 1 1]);
    else
        set(h,'FaceColor',[.75 .75 .75],'EdgeColor',[.25 .25 .25]);
    end;
end;
a = [];
a = plot(1:size(M,1),M(:,2),'r');    
for it = 1:size(M,1)
    if lab(it)==3
        %plot(it,M(it,2),'r.');
    end;
end;
ylabel(hdr{2});
xlabel('data samples');
axis('tight');
axes('Position',get(gca,'Position'));
[ax,h1,h2] = plotyy(1:size(M,1),M(:,1),1:size(M,1),M(:,3));
axis(ax,'tight');
axis(ax,'off');
set(h1,'LineWidth',3,'Color','y');
set(h2,'LineWidth',3,'Color','m');
legend([a,h1,h2],'time','file','var1');
subplot(423);
hold on;
bar(x2,n2);
plot([x2(1) x2(end)],[100 100],'r--');
set(gca,'XTick',sort(unique(M(1:1000:size(M,1),2))));
set(gca,'XTickLabel',sort(unique(M(1:1000:size(M,1),2))));
set(gca,'XTickLabelRotation',45);
xlabel(hdr{2});
ylabel('count');
xlim([min(M(:,2)) max(M(:,2))]);
subplot(424);
plot(diff(seg),'b-o');
set(gca,'XTick',1:5:length(unique(M(:,2))));
set(gca,'XTickLabelRotation',45);
xlabel('time');
ylabel('data samples x time point');
xlim([1 length(unique(M(:,2)))]);

subplot(425);
hold on;
bar(x1,mean(n1,1));
for it = 1:size(n1,1)
    for jt = 1:size(n1,2)
        plot(x1,n1(it,:),'ro');
    end;
end;
ylabel('count');
set(gca,'XTick',1:length(tpID));
set(gca,'XTickLabel',tpID);

subplot(426);
[R,p] = corr(M(:,3:end),'Type','Spearman');
[ixR,~] = ind2sub(size(R),find(R.*(p<0.05)>0.8));
ixR = sort(ixR);
imagesc(R.*(p<0.05));
caxis([-1 1]);
set(gca,'XTick',1:9);
set(gca,'YTick',1:9);
set(gca,'XTickLabel',hdr(3:end-1));
set(gca,'YTickLabel',hdr(3:end-1));
set(gca,'XTickLabelRotation',45);
cb = colorbar;
set(get(cb,'YLabel'),'String','correlation');

subplot(427);
n = zeros(length(labId));
for it = 1:length(labId)
    ix = find(lab == labId(it));    
    e = [];cnt = 0;
    for jt = 1:length(ix)   
        if ix(jt)-1 >1
            cnt = cnt+1;
            e(cnt) = lab(ix(jt)-1);
        end;
    end;
    n(:,labId(it)) = hist(e,labId)./length(ix);
end;
imagesc(labId,labId,n);
set(gca,'XTick',labId);
set(gca,'YTick',labId);
set(gca,'XTickLabel',tpID);
set(gca,'XTickLabelRotation',45);
set(gca,'YTickLabel',tpID);
%set(gca,'YTickLabelRotation',45);
cb = colorbar;
set(get(cb,'YLabel'),'String','Tranisition probability');

subplot(428);
segIx = find(sign([0;diff(M(:,3))])==-1);
sIx = 1:segIx(1);
xc = {};
lag = {};
t = M(:,2);
tId = unique(t);
n = zeros(length(labId),length(tId));
for kt = 1:length(labId)
    eix = find(lab == labId(kt));
    n(kt,:) = hist(t(eix),tId);
end;

[xc,lag] = xcorr(n','coeff');

hold on;
plot(lag,mean(xc(:,[2:5]),2));
plot(lag,mean(xc(:,[6 8:10]),2));
plot(lag,mean(xc(:,[11:12 14:15]),2));
plot(lag,mean(xc(:,[16:18 20]),2));
plot(lag,mean(xc(:,[21:24]),2));

plot([0 0],[min(min([xc])) max(max([xc]))],'k--','LineWidth',3);

xlabel('Lag [time]');
ylabel('average cross-correlation');
axis tight;

%%
figure;
subplot(231);
labId = unique(lab);
m = [];
se = [];
for it = 1:length( labId )
    ix = find(lab == labId(it));
    m(it,:) = mean(M(ix,3:end-1),1);
    se(it,:) = std(M(ix,3:end-1),0,1)./sqrt(length(ix)-1);
end;

hold on;
h = [];
%h = errorbar(labId*ones(1,size(m,2)),m,[],m+se,'s-');
h = plot(labId,m,'s-');
for it = 1:length(ixR)
    set(h(ixR(it)),'LineWidth',3);
end;
legend(h,hdr(3:end-1));
set(gca,'XTick',labId);
xlim([labId(1)-1 labId(end)+1]);
xlabel('type');
ylabel('average');

subplot(232);
F = zeros(size(M(:,3:end-1),2),1);
cnt = 0;
for it = 3:size(M,2)-1
    cnt = cnt+1;
    [p,table,stats] = anova1(M(:,it),lab,'off');
    F(cnt) = table{2,5};
end;
bar(1:length(F),F);
set(gca,'XTick',1:length(F));
set(gca,'XTickLabel',hdr(3:end-1));
set(gca,'XTickLabelRotation',45);
ylabel('F-value');

subplot(233);
[ix1,ix2] = ind2sub(size(R),find(R.*(p<0.05)>0.8));
ix1 = sort(ix1);

labId = unique(lab);
m = [];
for it = 1:length( labId )
    ix = find(lab == labId(it));
    m(it,:) = mean(M(ix,3:end-1),1);
end;

hold on;
h = [];
for it = 1:length(labId)  
    h(it) = area([labId(it)-.5 labId(it)+.5],ones(1,2)*max(max(m)),min(min(m)));
end;
for it = 1:length(C)
    set(h([it]),'FaceColor',C{it},'EdgeColor',C{it},'ShowBaseLine','off');
end;

selIx = [find(strcmp(hdr,'var4')) find(strcmp(hdr,'var5')) find(strcmp(hdr,'var7')) find(strcmp(hdr,'var8')) find(strcmp(hdr,'var9'))];
h = [];
h = plot(labId,m(:,selIx-2),'ks-','LineWidth',3);

set(gca,'XTick',labId);
xlim([labId(1)-1 labId(end)+1]);
xlabel('type');
ylabel('average');
axis tight;

subplot(234);
[COEFF, SCORE] = pca(M(:,[6:7 9:11]));
[~,sIx] = sort(diag(COEFF));


imagesc(COEFF);axis xy;caxis([-1 1]);
set(gca,'XTick',1:size(COEFF,2));
set(gca,'YTick',1:size(SCORE,2));
set(gca,'YTickLabel',hdr([6:7 9:11]));
xlabel('PC #');
cb = colorbar;
set(get(cb,'YLabel'),'String','PC weights');

subplot(235);
hold on;
for it = 1:length( labId )
    ix = find(lab == labId(it));
    plot(((SCORE(ix,1))),((SCORE(ix,2))),'o','Color',C{it});
end;
xlabel('PC1');ylabel('PC2');
axis tight;

subplot(236);
hold on;
labId = unique(lab);
h = [];
for it = 1:length( labId )
    ix = find(lab == labId(it));
    h(it) =plot3(SCORE(ix,1),SCORE(ix,2),SCORE(ix,4),'o','Color',C{it});
end;
legend(h,'type1','type2','type3','type4','type5');
xlabel('PC1');ylabel('PC2');zlabel('PC4');
view([-197 38]);
grid on;
axis tight;

%%
selIx = [find(strcmp(hdr,'var4')) find(strcmp(hdr,'var5')) find(strcmp(hdr,'var7')) find(strcmp(hdr,'var8')) find(strcmp(hdr,'var9'))];

P = zeros(length(labId),1);
for it = 1:length(labId)
    ix = find(lab == labId(it));
    P(it) = length(ix)/length(lab);
end;

P2 = [length(find(lab==1)) length(find(lab~=1))]./length(lab);

n = size(M,1);
rIx = randperm(n);

trIx = rIx(1:floor(n*0.9));
rIx(ismember(rIx,trIx)) = [];

rIx = rIx(randperm(length(rIx)));
teIx = rIx(1:floor(n*0.1));
rIx(ismember(rIx,teIx)) = [];

%% linear discriminant       
cnt = 0;
errTr = zeros(length(5e3:5e2:length(trIx)),2);
errCv = zeros(length(5e3:5e2:length(trIx)),2);

for it = 5e3:5e2:length(trIx)
    cnt = cnt+1;
    
    rIx = trIx(randperm(length(trIx)));
    lIx = rIx(1:it);
    
    D1 = [];
    D1 = fitcdiscr(SCORE(lIx,1:2),nominal(lab(lIx)==1),'DiscrimType','linear');
    
    D3 = [];
    D3 = fitcdiscr(SCORE(lIx,1:2),nominal(lab(lIx)==1),'DiscrimType','quadratic');      
    
    errTr(cnt,1) = resubLoss(D1);    
    errTr(cnt,2) = resubLoss(D3);
    
    cvmodel = crossval(D1,'Holdout',.1);
    errCv(cnt,1) = kfoldLoss(cvmodel);
    
    cvmodel = crossval(D3,'Holdout',.1);
    errCv(cnt,2) = kfoldLoss(cvmodel);

end;

%%
figure;
subplot(321);
hold on;
h = [];
h(1)=plot(5e3:5e2:length(trIx),errTr(:,1),'b--');
h(2)=plot(5e3:5e2:length(trIx),errCv(:,1),'r');
legend(h,'Training','Validation');
%ylim([0 1]);
xlabel('Number of training samples');
ylabel('Error');
xlim([5e3 length(trIx)]);
%ylim([.75 1]);
title('Model 1: linear binary');
 
subplot(322);
hold on;
h = [];
h(1)=plot(5e3:5e2:length(trIx),errTr(:,2),'b--');
h(2)=plot(5e3:5e2:length(trIx),errCv(:,2),'r');
%ylim([0 1]);
xlabel('Number of training samples');
ylabel('Error');
xlim([5e3 length(trIx)]);
%ylim([.75 1]);
title(['Model 2: quadratic binary']);

subplot(323);
hold on;
plot(SCORE(lab~=1,1),SCORE(lab~=1,2),'bo');
plot(SCORE(lab==1,1),SCORE(lab==1,2),'rx');
K = D1.Coeffs(1,2).Const;
L = D1.Coeffs(1,2).Linear;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f,[min(SCORE(:,1)) max(SCORE(:,1)) min(SCORE(:,2)) max(SCORE(:,2)) ]);
h2.Color = 'g';
h2.LineWidth = 2;
xlabel(gca,'PC1');
ylabel(gca,'PC2');

subplot(324);
hold on;
plot(SCORE(lab~=1,1),SCORE(lab~=1,2),'bo');
plot(SCORE(lab==1,1),SCORE(lab==1,2),'rx');
K = D3.Coeffs(1,2).Const;
L = D3.Coeffs(1,2).Linear;
Q = D3.Coeffs(1,2).Quadratic;

f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = ezplot(f,[min(SCORE(:,1)) max(SCORE(:,1)) min(SCORE(:,2)) max(SCORE(:,2)) ]);
h2.Color = 'g';
h2.LineWidth = 2;
xlabel(gca,'PC1');
ylabel(gca,'PC2');

subplot(325);
R1 = confusionmat(D1.Y,resubPredict(D1));
R2 = confusionmat(D3.Y,resubPredict(D3));
R3 = confusionmat(nominal(lab(teIx)==1),predict(D1,SCORE(teIx,1:2)));
R4 = confusionmat(nominal(lab(teIx)==1),predict(D3,SCORE(teIx,1:2)));

fp1 = R1(2,1)/sum(R1(2,:));
tp1 = R1(1,1)/sum(R1(1,:));
fp2 = R2(2,1)/sum(R2(2,:));
tp2 = R2(1,1)/sum(R2(1,:));
fp3 = R3(2,1)/sum(R3(2,:));
tp3 = R3(1,1)/sum(R3(1,:));
fp4 = R4(2,1)/sum(R4(2,:));
tp4 = R4(1,1)/sum(R4(1,:));

hold on;
plot(fp1,tp1,'bx','LineWidth',3);
plot(fp2,tp2,'rx','LineWidth',3);
plot(fp3,tp3,'bo','LineWidth',3,'MarkerSize',8);
plot(fp4,tp4,'ro','LineWidth',3,'MarkerSize',8);
plot([0 1],[0 1]);
xlabel(gca,'False positive rate');
ylabel(gca,'True positive rate');

clear D1 D3 R1 R2;

%% multiclass support vector machine
t = templateSVM('Standardize',1,'KernelFunction','rbf');
options = statset('UseParallel',true);

errTr = [];
errCv = [];
cnt = 0;
for it = 5e3:5e2:length(trIx)
    
    cnt = cnt+1;
    rIx = trIx(randperm(length(trIx)));
    lIx = rIx(1:it);
    
    svm = fitcecoc(SCORE(lIx,1:2),tp(lIx),'Learners',t,'FitPosterior',1,...
        'ClassNames',tpID,'Verbose',2,'Options',options);
    
    errTr(cnt) = resubLoss(svm);
    cvmodel = crossval(svm,'Holdout',.1);
    errCv(cnt) = kfoldLoss(cvmodel);
end;

[label,~,~,Posterior] = resubPredict(svm,'Verbose',1);


xMax = max(SCORE(:,1:2));
xMin = min(SCORE(:,1:2));

x1Pts = linspace(xMin(1),xMax(1));
x2Pts = linspace(xMin(2),xMax(2));
[x1Grid,x2Grid] = meshgrid(x1Pts,x2Pts);

[~,~,~,PosteriorRegion] = predict(svm,[x1Grid(:),x2Grid(:)]);

%%
figure;
subplot(221);
hold on;
plot(5e3:5e2:length(trIx),errTr,'b--');
plot(5e3:5e2:length(trIx),errCv,'r');
xlabel('Number of training samples');
ylabel('Error');

subplot(222);
contourf(x1Grid,x2Grid,...
        reshape(max(PosteriorRegion,[],2),size(x1Grid,1),size(x1Grid,2)));

colormap bone;
h = colorbar;
h.YLabel.String = 'Maximum posterior';
h.YLabel.FontSize = 15;
hold on
gh = gscatter(SCORE(teIx,1),SCORE(teIx,2),lab(teIx),'rygcb','o',8);
gh(2).LineWidth = 2;
gh(3).LineWidth = 2;

title 'Test data and Maximum Posterior';
xlabel 'PC1';
ylabel 'PC2';
axis tight
legend(gh,'Location','NorthWest')
hold off

subplot(223);
R1 = confusionmat(svm.Y,resubPredict(svm));
R2 = confusionmat(tp(teIx),predict(svm,SCORE(teIx,1:2)));
tp1 = diag(R1)./sum(R1,2);
fp1 = zeros(size(R1,1),1);
for it = 1:size(R1,2)
    ix = setdiff(1:size(R1,1),it);
    fp1(it) = sum(R1(ix,it))./sum(R1(:,it));
end;
tp2 = diag(R2)./sum(R2,2);
fp2 = zeros(size(R2,1),1);
for it = 1:size(R2,2)
    ix = setdiff(1:size(R2,1),it);
    fp2(it) = sum(R2(ix,it))./sum(R2(:,it));
end;
hold on;
for it = 1:length(tp1)
    plot(fp1(it),tp1(it),'x','Color',C{it},'LineWidth',3);
    plot(fp2(it),tp2(it),'o','Color',C{it},'LineWidth',3,'MarkerSize',12);
end;
plot([0 1],[0 1]);
xlabel(gca,'False positive rate');
ylabel(gca,'True positive rate');

%%
tTree = templateTree('surrogate','off');
tEnsemble = templateEnsemble('RUSBoost',300,tTree);

pool = parpool; % Invoke workers
options = statset('UseParallel',true);
Mdl = fitcecoc(SCORE(trIx,1:2),tp(trIx),'Coding','onevsall','Learners',tEnsemble,...
                'Prior','uniform','Options',options);
         
cv = crossval(Mdl,'Options',options);
yhat = kfoldPredict(cv,'Options',options);
ConfMat = confusionmat(tp(trIx),yhat);

