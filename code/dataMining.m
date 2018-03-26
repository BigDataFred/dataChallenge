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

%%
M = str2double(bigD(:,1:end-1));
tp = bigD(:,end);

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

[n1,x1] = hist(lab,[1:length(tpID)]);
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
bar(x1,n1);
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
set(get(cb,'YLabel'),'String','probability');

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
ylabel('F-values');

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

selIx = [find(strcmp(hdr,'var2')) find(strcmp(hdr,'var3')) find(strcmp(hdr,'var4')) find(strcmp(hdr,'var5')) find(strcmp(hdr,'var7')) find(strcmp(hdr,'var8')) find(strcmp(hdr,'var9'))];
h = [];
h = plot(labId,m(:,selIx-2),'ks-','LineWidth',3);

set(gca,'XTick',labId);
xlim([labId(1)-1 labId(end)+1]);
xlabel('type');
ylabel('average');
axis tight;

subplot(234);
[COEFF, SCORE] = pca(M(:,3:end-1));
imagesc(COEFF);axis xy;caxis([-1 1]);
set(gca,'XTick',1:size(COEFF,2));
set(gca,'YTick',1:size(SCORE,2));
set(gca,'YTickLabel',hdr(3:end-1));
xlabel('PC #');
cb = colorbar;
set(get(cb,'YLabel'),'String','PC weights');

subplot(235);
hold on;
labId = unique(lab);
h = [];
for it = 1:length( labId )
    ix = find(lab == labId(it));
    h(it) =plot3(SCORE(ix,1),SCORE(ix,2),SCORE(ix,4),'o','Color',C{it});
end;
legend(h,'type1','type2','type3','type4','type5');
xlabel('PC1');ylabel('PC2');zlabel('PC4');
view([-166 7]);
grid on;

%%
X = [SCORE];
n = size(X,1);
rIx = randperm(n);

training = rIx(1:floor(n*0.8));
rIx(ismember(rIx,training)) = [];

rIx = rIx(randperm(length(rIx)));
crossv = rIx(1:floor(n*0.1));
rIx(ismember(rIx,crossv)) = [];

rIx = rIx(randperm(length(rIx)));
test = rIx(1:floor(n*0.1));
rIx(ismember(rIx,test)) = [];

figure;
subplot(221);
hold on;

errRate = [];
pcIx =[];
for jt = 1:size(X,2)
    pcIx = [pcIx jt]
    for it = 1:5
        
        Y = nominal(ismember(lab,labId(it)));
        D = [];
        D = fitcdiscr(X(training,pcIx),Y(training),'DiscrimType','diagLinear','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','Prior','bayesopt'));
        
        CM1 = confusionmat(Y(crossv),predict(D,X(crossv,pcIx)));
        
        errRate(jt,it) = sum(Y(crossv)~= predict(D,X(crossv,pcIx)))/length(crossv);
    end;
    
end;
hold on;
contour(errRate);
plot(1:length(tpID),5*ones(1,length(tpID)),'r--');
axis xy;
set(gca,'YTick',1:size(X,2));
set(gca,'XTick',1:length(tpID));

ylabel('number of PCs included');
xlabel('type');

cb = colorbar;
set(get(cb,'YLabel'),'String','Test-error (%)');

subplot(222);
pcIx = 1:5;
hold on;
perfCV = [];perfTE = [];
D = {};
for it = 1:5
    
    Y = nominal(ismember(lab,labId(it)));
    
    D{it} = fitcdiscr(X(training,pcIx),Y(training),'DiscrimType','diagLinear','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','Prior','bayesopt'));
    
%     CM1 = confusionmat(Y(crossv),predict(D{it},X(crossv,pcIx)));
%     perfCV(it,:) = CM1(:)./length(crossv);
    CM2 = confusionmat(Y(test),predict(D{it},X(test,pcIx)));
    perfTE(it,:) = CM2(:)./length(test);
    
end;

h = [];
for it = 1:size(perfTE,1)    
    %plot(perfCV(it,2),perfCV(it,1),'o','Color',C{it},'MarkerSize',8);
    h(it) = plot(perfTE(it,2),perfTE(it,1),'o','Color',C{it},'MarkerSize',16);
end;
legend(h,tpID);
title('Naive bayesian classifier');
ylabel('True positives (%)');
xlabel('False positives (%)');

%%
x =M(:,selIx);

for it = 1:size(x,2)
    x(:,it) = (x(:,it)-mean(x(:,it)))./std(x(:,it));
end;

figure;
subplot(211);
hold on;
h = [];
h = plot(M(:,2),x);
legend(h,hdr(selIx));
plot(M([1 size(M,1)],2),[-4 -4],'Color',[.75 .75 .75]);
plot(M([1 size(M,1)],2),[4 4],'Color',[.75 .75 .75]);
plot(M(find(x(:,5)>4),2),x(find(x(:,5)>4),5),'ro');
%xlim([15000 17000]);
ylim([min(min(x))-1 max(max(x))+1]);
ylabel('z-score');
xlabel('data samples');
subplot(212);
hold on;
plot(M(:,2),M(:,end));
for it = 1:length(labId)
    plot(M(find(lab==labId(it)),2),M(find(lab==labId(it)),end),'o','Color',C{it});
end;
ylim([0 6]);
%xlim([15000 17000]);
xlabel('data samples');
ylabel('type');

%%
figure;
h = [];
for it = 1:length(labId)
    
    subplot(331);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,1),x(ix,2),'o','Color',C{it});
    xlabel(hdr(selIx(1)));
    ylabel(hdr(selIx(2)));
    subplot(332);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,3),x(ix,4),'o','Color',C{it});
    xlabel(hdr(selIx(3)));
    ylabel(hdr(selIx(4)));
    subplot(333);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,6),x(ix,7),'o','Color',C{it});
    xlabel(hdr(selIx(6)));
    ylabel(hdr(selIx(7)));
    subplot(334);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,3),x(ix,6),'o','Color',C{it});
    xlabel(hdr(selIx(3)));
    ylabel(hdr(selIx(6)));
    subplot(335);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,3),x(ix,7),'o','Color',C{it});
    xlabel(hdr(selIx(3)));
    ylabel(hdr(selIx(7)));
    subplot(337);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,4),x(ix,6),'o','Color',C{it});
    xlabel(hdr(selIx(4)));
    ylabel(hdr(selIx(6)));
    subplot(338);
    hold on;
    ix = find(lab == labId(it) );    
    h(it) = plot(x(ix,4),x(ix,7),'o','Color',C{it});
    xlabel(hdr(selIx(4)));
    ylabel(hdr(selIx(7)));
end;
legend(h,tpID);













