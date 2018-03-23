%%
p2d = '/home/adf/rouxf/Downloads/data-lover-challenge/csvDat/';
files = dir([p2d,'*.csv'])

%%
for it = 1
        
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
    
end;

%%
M = str2double(X(:,1:end-1));
tp = X(:,end);

%%
tpID = unique(tp);
lab = zeros(length(tp),1);
for it = 1:length(tpID)
    lab(find(strcmp(tp,tpID(it)))) = it;
end;
M(:,end+1) = lab;

seg = [1;find([0;diff(M(:,2))]==1);size(M,1)];

%%
[n1,x1] = hist(lab,[1:length(tpID)]);
[n2,x2] = hist(M(:,2),sort(unique(M(:,2))));
figure;
subplot(221);
bar(x1,n1);
ylabel('count');
set(gca,'XTick',1:length(tpID));
set(gca,'XTickLabel',tp);
subplot(222);
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
plot(1:size(M,1),M(:,2),'y');    
ylabel(hdr{2});
axis('tight');
subplot(223);
[xc,lag] = xcorr(M(:,end)-mean(M(:,end)),500,'coeff');
plot(lag,xc);
axis tight;
subplot(224);
hold on;
bar(x2,n2);
plot([x2(1) x2(end)],[100 100],'r--');
set(gca,'XTick',sort(unique(M(1:1000:size(M,1),2))));
set(gca,'XTickLabel',sort(unique(M(1:1000:size(M,1),2))));
xlabel(hdr{2});
ylabel('count');
xlim([min(M(:,2)) max(M(:,2))]);

%%
[R,p] = corr(M(:,3:end));

figure;
subplot(121);
imagesc(R);
caxis([-1 1]);
set(gca,'XTick',1:9);
set(gca,'YTick',1:9);
set(gca,'XTickLabel',hdr(3:end-1));
set(gca,'YTickLabel',hdr(3:end-1));
subplot(122);
imagesc(R.*(p<0.05));
caxis([-1 1]);
set(gca,'XTick',1:9);
set(gca,'YTick',1:9);
set(gca,'XTickLabel',hdr(3:end-1));
set(gca,'YTickLabel',hdr(3:end-1));