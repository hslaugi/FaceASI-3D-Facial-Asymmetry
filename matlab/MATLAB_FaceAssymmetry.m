function Results = MATLAB_FaceAssymmetry(varargin)
% Created: 2026-10-20T18:25:15-05:00 (America/Chicago).
% Run MATLAB_FaceAssymmetry to select a scan CSV.
% FACIALEXPERIMENT_3D_ASI_DENSEEYELID  Three cloud modes, optional dense eyelid/eyebrow vertices and responsive ASI moving mean.
% Three camera-frame display modes: solid color, recorded RGB, and depth.
% RGB/depth modes overlay all available 3D landmark dots (up to 21).
% Missing XYZ remains missing; modes never generate or infer landmarks.
% ALL measurement lines are direct A-to-B Euclidean segments, never unrelated midline projections.
% New: paired dropdowns (all 21 recorded labels), per-frame distance/SD, N-frame trailing means,
% selected-point colors, large measurement plot and synchronized 3D highlights.
% Dense annotations require separately recorded zero-based MediaPipe mesh
% vertices from the single FaceASI scan CSV (dense_landmark rows).
% No embedded face recording is distributed in this public version.
% Optional DatasetMAT with denseIDs/denseXYZ populates dense overlays.
% Eyelid aperture = abs(upper-lower difference along the current face V-axis),
% a provisional *eyelid opening* proxy, NOT clinical ptosis/MRD1.
% Moving N-frame ASI mean is displayed when selected; raw ASI retained.
% See docs/REPRODUCIBILITY.md for testing and research limitations.
% Public release: load your authorized RGB/XYZ scan CSV (no bundled face data).
% Use the 3D dropdown in the playback controls to switch display styles.
% The 21 label IDs are retained, but a missing landmark (NaN or failed snap)
% cannot be plotted; the displayed count reflects available XYZ only.
%
% Run:
%   T = MATLAB_FaceAssymmetry; % select a device CSV
%   T = MATLAB_FaceAssymmetry('ScanCSV','Scan_001_Neutral.csv');
%   % UseEmbedded demo unavailable in this privacy-safe public release.
%   T = MATLAB_FaceAssymmetry('FPS', measuredOriginalFPS);
%   T = MATLAB_FaceAssymmetry('FrameProvider',@yourProvider);
%
% FrameProvider() returns struct with .cloud (Nx3/Nx6, mm),
% .landmarks (21x3 indexed by embeddedData.landmarkIDs) and preferably
% .timestampSeconds. It MUST deliver anatomically labeled landmarks.
% A missing detector cannot be replaced by point-cloud proximity matching.
%
% Bilateral asymmetry for one pair i, in head-oriented U,V,W coordinates:
% hL=|U_L|; hR=|U_R|; width=hL+hR
% dH=|hL-hR|; dV=|V_L-V_R|; dD=|W_L-W_R|.
% The reference sagittal axis U=0 is fixed by NASION -> SUBNASALE;
% it no longer depends on lip/chin points. Head pose is only partly removed.
% For selected valid pairs in region G:
% H = 100*SUM(dH)/SUM(width), V = 100*SUM(dV)/SUM(width),
% D = 100*SUM(dD)/SUM(width)
% BilateralASI = sqrt(H^2+V^2+D^2).
% Optional selected midline points add separate M =
%   100*SUM(2*abs(U_midlinePoint))/SUM(selected pair widths).
% The factor 2 compares offset of a single midline point to a hypothetical
% symmetric reflection; it is a design choice, NOT a clinical standard.
% Selected TOTAL index = sqrt(H^2+V^2+D^2+M^2).
% Overall = pooled numerators/denominators, not average of region ASIs.
% Selecting different landmarks changes the meaning and comparability.
% Midline anchors NASION / SUBNASALE define the reference; their measured
% U=0 by construction and must not be counted as independent evidence.
%
% Offline stats are CUMULATIVE through the displayed frame, sample SD
% N-1; no finite values or <2 frames gives SD=N/A. Missing data is NaN.
% Velocity: v_i(k)=(q_i(k)-q_i(k-1))/dt, RMS speed =
% sqrt(mean_i(sum(v_i.^2))). Delta-velocity is the finite difference
% between vectors divided by dt; NOT clinical nerve/muscle speed.
% FPS is acquisition frame rate only when externally verified. Replay
% speed does NOT change acquisition time. Without known timing, motion
% is mm/frame and delta-velocity mm/frame^2.
%
% Landmarks are supplied seeds snapped to the original cloud: this is
% NOT independent anatomical correctness validation. Outlier warnings and
% optional CAUSAL 3-frame median smoothing do not create ground truth.
% Individual graph points are distances/coordinates (mm), not single-point ASI.
% Each dot may be inspected with a data cursor or a full values table.
% MATLAB R2020b+; base MATLAB only. No camera driver included.

    ip = inputParser;
    addParameter(ip,'FrameProvider',[],@(v) isempty(v)||isa(v,'function_handle'));
    addParameter(ip,'FPS',NaN,@(v) isnumeric(v)&&isscalar(v)&&(isnan(v)||(isfinite(v)&&v>0)));
    addParameter(ip,'SnapRadius_mm',3,@(v) isnumeric(v)&&isscalar(v)&&isfinite(v)&&v>0);
    addParameter(ip,'UpdatePeriod_s',0.18,@(v) isnumeric(v)&&isscalar(v)&&isfinite(v)&&v>=0.05);
    addParameter(ip,'MaxLiveFrames',120,@(v) isnumeric(v)&&isscalar(v)&&isfinite(v)&&v>=10&&v==floor(v));
    addParameter(ip,'ScanCSV','',@(v) ischar(v)||isstring(v));
    addParameter(ip,'UseEmbedded',false,@islogical);
    addParameter(ip,'ImportOnly',false,@islogical);
    addParameter(ip,'DatasetMAT','',@(v) ischar(v)||isstring(v));
    parse(ip,varargin{:}); opt = ip.Results;

    % Run without arguments to select a new device CSV; no editing paths in code.
    if isempty(char(opt.ScanCSV)) && isempty(char(opt.DatasetMAT)) && isempty(opt.FrameProvider) && ~opt.UseEmbedded
        lastFolder=getpref('FaceASI','LastScanFolder',pwd);
        if ~isfolder(lastFolder),lastFolder=pwd;end
        [fn,fp]=uigetfile('*.csv','Open one FaceASI scan CSV',lastFolder);
        if isequal(fn,0),Results=table();return;end
        opt.ScanCSV=fullfile(fp,fn);setpref('FaceASI','LastScanFolder',fp);
    end
    hasCSV=~isempty(char(opt.ScanCSV));deviceFrames=table();deviceSummary=table();
    if hasCSV && (~isempty(char(opt.DatasetMAT)) || ~isempty(opt.FrameProvider))
        error('Choose only one of ScanCSV, DatasetMAT or FrameProvider.');
    end
    D = embeddedData(); IDs = D.landmarkIDs; nLM = numel(IDs);
    pairNames = {'Superciliare (proxy)';'Lateral canthus';'Medial canthus';'Alare';'Cheilion';'Zygomatic arch'};
    landmarkShortNames = {'Alare left';'Alare right';'Cheilion left';'Cheilion right'; ...
        'Forehead proxy';'Glabella';'Gnathion';'Labiale inferius (LI)'; ...
        'Labiale superius (LS)';'Lateral canthus left';'Lateral canthus right'; ...
        'Medial canthus left';'Medial canthus right';'Menton';'Pronasale'; ...
        'Sellion / Nasion';'Subnasale';'Superciliare left (proxy)'; ...
        'Superciliare right (proxy)';'Zygoma left';'Zygoma right'};
    landmarkMetricOptions = {'Distance to midline |U| (mm)'; ...
        'Signed U (mm)';'Vertical V (mm)';'Depth W (mm)'; ...
        'Camera X (mm)';'Camera Y (mm)';'Camera Z (mm)'; ...
        '3D step since previous frame (mm)'};
    pairL = idList({'SUPERCILIARE_L','LATERAL_CANTHUS_L','MEDIAL_CANTHUS_L','ALARE_L','CHEILION_L','ZYGOMA_L'});
    pairR = idList({'SUPERCILIARE_R','LATERAL_CANTHUS_R','MEDIAL_CANTHUS_R','ALARE_R','CHEILION_R','ZYGOMA_R'});
    upperIdx = 1:3; lowerIdx = 4:6;
    selectedPairs = logical([1 1 1 1 1 0]); % ZYGOMA_L absent; may be selected for QC
    upperMid = idList({'FOREHEAD_PROXY','GLABELLA','SELLION_NASION'});
    lowerMid = idList({'PRONASALE','SUBNASALE','LABIALE_SUPERIUS', ...
        'LABIALE_INFERIUS','GNATHION','MENTON'});
    selectedMid = false(1,nLM);
    % LS and LI opt-in via Select landmarks: suspicious LI position in capture.
    trackIDs = unique([pairL(1:5) pairR(1:5)]);
    midIDs = idList({'GLABELLA','SELLION_NASION','SUBNASALE'});
    idNasion = idList({'SELLION_NASION'}); idSubnasale = idList({'SUBNASALE'});

    hasProvider = ~isempty(opt.FrameProvider);
    if hasProvider && ~isempty(char(opt.DatasetMAT))
        error('Use either FrameProvider or DatasetMAT, not both.');
    end
    if hasCSV
        S=readScanCSV(char(opt.ScanCSV),IDs);
        if opt.ImportOnly,Results=S;return;end
        rawSeeds=S.xyz;rawClouds=S.clouds;acquiredTime=S.timestamps;
        deviceFrames=S.deviceFrames;deviceSummary=S.deviceSummary;
        frameSource=repmat({'Device CSV: original measured XYZ'},numel(rawClouds),1);
    elseif hasProvider
        rawClouds = cell(0,1); rawSeeds = nan(0,nLM,3);
        acquiredTime = nan(0,1); frameSource = cell(0,1);
    elseif ~isempty(char(opt.DatasetMAT))
        S=load(char(opt.DatasetMAT));
        % Only explicit anatomical labels are accepted. No landmark fabrication.
        if ~isfield(S,'landmarkIDs') || ~isfield(S,'xyz') || ~isfield(S,'clouds')
            error('DatasetMAT must contain landmarkIDs, xyz [N x 21 x 3], and clouds {N x 1}.');
        end
        labelIDs=cellstr(string(S.landmarkIDs(:)));
        if numel(labelIDs)~=nLM || ~isequal(labelIDs,IDs(:))
            error('Dataset landmarkIDs must be in the same 21-ID order as the embedded source.');
        end
        rawSeeds=double(S.xyz);rawClouds=S.clouds(:);
        if size(rawSeeds,2)~=nLM || size(rawSeeds,3)~=3 || ...
                size(rawSeeds,1)~=numel(rawClouds)
            error('Dataset size mismatch: xyz must be Nx21x3 and clouds Nx1.');
        end
        acquiredTime=nan(size(rawSeeds,1),1);
        if isfield(S,'timestamps')
            ts=double(S.timestamps(:));
            if numel(ts)~=numel(rawClouds),error('timestamps length mismatch');end
            acquiredTime=ts;
        end
        frameSource=repmat({'External labeled MAT recording'},numel(rawClouds),1);
        for f=1:numel(rawClouds)
            a=rawClouds{f};
            if ~isnumeric(a)||~ismatrix(a)||size(a,2)<3
                error('Dataset clouds{%d} must be a numeric Nx3 or Nx6 matrix.',f);
            end
        end
    else
        R = embeddedRawFrames(); rawClouds = R.clouds(:,1);
        rawSeeds = D.xyz{1}; acquiredTime = nan(size(rawSeeds,1),1);
        frameSource = repmat({'Exported capture labels'},size(rawSeeds,1),1);
    end
    % Curated zero-based MediaPipe vertices. Shared contract with FaceASI v52.
    denseIDs=[33 7 163 144 145 153 154 155 133 173 157 158 159 160 161 246 362 382 381 380 374 373 390 249 263 466 388 387 386 385 384 398 70 63 105 66 107 55 65 52 53 46 336 296 334 293 300 285 295 282 283 276 1 2 4 5 6 19 94 97 98 168 195 197 326 327];
    denseXYZ=nan(size(rawSeeds,1),numel(denseIDs),3);
    if hasCSV
        denseIDs=S.denseIDs;denseXYZ=S.denseXYZ;
    elseif ~hasProvider && ~isempty(char(opt.DatasetMAT)) && isfield(S,'denseXYZ')
        if ~isfield(S,'denseIDs') || ~isequal(double(S.denseIDs(:)'),denseIDs)
            error('Dataset denseIDs must match the curated FaceASI v52 mesh ID order.');
        end
        denseValues=double(S.denseXYZ);
        if ~isequal(size(denseValues),size(denseXYZ))
            error('denseXYZ must be Nx%d x3, aligned to the 21-landmark frames.',numel(denseIDs));
        end
        denseXYZ=denseValues;
    end
    % Mesh landmark groups contain 3D coordinates only where per-vertex depth
    % passed the same-frame sensor sampling checks. Missing values remain NaN.
    denseEyeIDs=[33 7 163 144 145 153 154 155 133 173 157 158 159 160 161 246 362 382 381 380 374 373 390 249 263 466 388 387 386 385 384 398];
    denseBrowIDs=[70 63 105 66 107 55 65 52 53 46 336 296 334 293 300 285 295 282 283 276];
    denseNoseIDs=[1 2 4 5 6 19 94 97 98 168 195 197 326 327];
    eyelidLeftUpper=386; eyelidLeftLower=374;
    eyelidRightUpper=159; eyelidRightLower=145;
    eyelidFig=[]; eyelidLines=gobjects(0,1); eyelidInfo=[];
    nFrames = size(rawSeeds,1); frame = 1; playing=false;
    fps = opt.FPS; snapRadius = opt.SnapRadius_mm; smoothMode=false;
    rawProcessed=[]; smoothProcessed=[];
    processed = nan(nFrames,nLM,3); localXYZ = nan(nFrames,nLM,3);
    measurements = repmat(emptyFrame(),nFrames,1);
    frameNumbers = (1:nFrames)'; totalReceived=nFrames;
    reviewFig=[];reviewMenu=[];reviewAxLR=[];reviewAxDelta=[];
    reviewText=[];reviewCurLR=[];reviewCurD=[];
    selectionA=idList({'GLABELLA'});
    selectionB=idList({'LATERAL_CANTHUS_L'});
    selectionColorA=[1.00 .57 .23];selectionColorB=[.75 .49 1.00];
    viewMode='measure';movingWindow=5;
    for k=1:nFrames
        [p,q,m] = processFrame(rawClouds{k},squeeze(rawSeeds(k,:,:)));
        processed(k,:,:) = reshape(p,[1 nLM 3]);
        localXYZ(k,:,:) = reshape(q,[1 nLM 3]); measurements(k)=m;
    end
    rawProcessed=processed; smoothProcessed=causalSmooth(rawProcessed);
    recomputeAll();

    upperColor=[.95 .26 .28]; lowerColor=[.30 .55 .98]; midColor=[.20 .82 .45];
    overallColor=[.77 .53 .96];
    % Colors are anatomical/visual: red upper, green middle, blue lower.
    % The original two-region experimental ASI includes alare and zygoma
    % (green midface) within its LOWER numerical score for compatibility.
    upperDisplayIDs=idList({'FOREHEAD_PROXY','GLABELLA', ...
        'SUPERCILIARE_L','SUPERCILIARE_R','LATERAL_CANTHUS_L', ...
        'LATERAL_CANTHUS_R','MEDIAL_CANTHUS_L','MEDIAL_CANTHUS_R'});
    midDisplayIDs=idList({'ALARE_L','ALARE_R','ZYGOMA_L','ZYGOMA_R', ...
        'PRONASALE','SELLION_NASION','SUBNASALE'});
    lowerDisplayIDs=idList({'CHEILION_L','CHEILION_R','LABIALE_SUPERIUS', ...
        'LABIALE_INFERIUS','GNATHION','MENTON'});
    fg=[.88 .94 .99]; bg=[.065 .087 .125]; panelBg=[.105 .135 .185];
    fig=figure('Name','3D Facial ASI | Experimental / Research Only', ...
        'NumberTitle','off','Color',bg,'Units','normalized', ...
        'OuterPosition',[.015 .04 .97 .91], ...
        'MenuBar','none','ToolBar','figure','CloseRequestFcn',@closeViewer);

    uicontrol(fig,'Style','text','Units','normalized','Position',[.022 .955 .60 .031], ...
        'String','MATLAB RECOMPUTED ASI | Original scores: Device results | RESEARCH ONLY', ...
        'BackgroundColor',bg,'ForegroundColor',fg,'FontSize',12,'FontWeight','bold', ...
        'HorizontalAlignment','left');
    uicontrol(fig,'Style','pushbutton','Units','normalized','Position',[.640 .955 .111 .035], ...
        'String','Open scan CSV','Callback',@openScanCSV);
    uicontrol(fig,'Style','pushbutton','Units','normalized','Position',[.758 .955 .104 .035], ...
        'String','Device results','Callback',@showDeviceResults);
    ax=axes('Parent',fig,'Units','normalized','Position',[.024 .292 .443 .637], ...
        'Color',[.03 .055 .085],'XColor',fg,'YColor',fg,'ZColor',fg,'Projection','perspective');
    hold(ax,'on'); grid(ax,'on'); axis(ax,'vis3d'); axis(ax,'equal'); set(ax,'YDir','reverse');
    xlabel(ax,'X camera (mm)');ylabel(ax,'Y camera (mm)');zlabel(ax,'Z depth (mm)');view(ax,0,90);
    cloudH=scatter3(ax,nan,nan,nan,7,[.43 .60 .67],'filled', ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',.76);
    marksU=scatter3(ax,nan,nan,nan,88,[.70 .77 .81],'filled','MarkerEdgeColor','w');
    marksL=scatter3(ax,nan,nan,nan,88,[.70 .77 .81],'filled','MarkerEdgeColor','w');
    marksM=scatter3(ax,nan,nan,nan,65,[.70 .77 .81],'filled','MarkerEdgeColor','w');
    marksOther=scatter3(ax,nan,nan,nan,38,[.70 .77 .81],'filled');
    % The A/B markers sit above all other landmarks. The displayed segment is
    % the ACTUAL Euclidean distance shown in the measurement plot.
    selectedMarkA=scatter3(ax,nan,nan,nan,220,selectionColorA,'filled', ...
        'MarkerEdgeColor','w','LineWidth',1.3,'Visible','off');
    selectedMarkB=scatter3(ax,nan,nan,nan,220,selectionColorB,'filled', ...
        'MarkerEdgeColor','w','LineWidth',1.3,'Visible','off');
    selectedLineA=plot3(ax,nan,nan,nan,'-','Color',selectionColorA, ...
        'LineWidth',3,'Visible','off');
    selectedLineB=plot3(ax,nan,nan,nan,'-','Color',selectionColorB, ...
        'LineWidth',3,'Visible','off');
    selectedDistanceLabel=text(ax,0,0,0,'','Color','w', ...
        'FontSize',10,'FontWeight','bold','BackgroundColor',[.10 .14 .19], ...
        'Visible','off','Clipping','on','Interpreter','none');
    % Dense *anatomically identified mesh vertices*, not generic cloud points.
    % Make layers small to preserve visibility of main A/B selections.
    denseEyeH=scatter3(ax,nan,nan,nan,22,[1 .77 .22],'filled', ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',.92,'Visible','off');
    denseBrowH=scatter3(ax,nan,nan,nan,21,[1 .38 .44],'filled', ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',.92,'Visible','off');
    denseNoseH=scatter3(ax,nan,nan,nan,19,[.28 .96 .70],'filled', ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',.9,'Visible','off');
    midLine=plot3(ax,nan,nan,nan,'--','Color',midColor,'LineWidth',1.5);
    allMidSlots=[upperMid(:)' lowerMid(:)'];
    midMeasLines=gobjects(numel(allMidSlots),1);
    midMeasText=gobjects(numel(allMidSlots),1);
    for j=1:numel(allMidSlots)
        midMeasLines(j)=plot3(ax,nan,nan,nan,'--', ...
            'Color',midColor,'LineWidth',1.25,'Visible','off');
        midMeasText(j)=text(ax,0,0,0,'','Color',midColor,'FontSize',8, ...
            'Interpreter','none','Visible','off','Clipping','on');
    end
    measLines=gobjects(6,4); measText=gobjects(6,2);
    for j=1:6
        c=pairTint(j);
        for z=1:4
            ls='-';if z>=3,ls=':';end
            measLines(j,z)=plot3(ax,nan,nan,nan,ls,'Color',c,'LineWidth',1.55);
        end
        for z=1:2
            measText(j,z)=text(ax,0,0,0,'','Color',c,'FontSize',8, ...
                'BackgroundColor',[.045 .07 .10],'Visible','off','Clipping','on');
        end
    end
    lmLabels=gobjects(nLM,1);
    for j=1:nLM
        lmLabels(j)=text(ax,0,0,0,'','Color',fg,'FontSize',8, ...
            'Interpreter','none','Visible','off','Clipping','on');
    end

    % Mouse-drag rotation acts on the 3D axes; toolbar remains available.
    % These explicit controls work even if MATLAB's default rotation is off.
    rotButton=uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.035 .929 .095 .026],'String','Rotate 3D', ...
        'Callback',@enableRotate);
    uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.135 .929 .068 .026],'String','Pan', ...
        'Callback',@enablePan);
    uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.208 .929 .068 .026],'String','Zoom', ...
        'Callback',@enableZoom);
    uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.281 .929 .087 .026],'String','Side view', ...
        'Callback',@sideView);
    uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.373 .929 .090 .026],'String','Front view', ...
        'Callback',@frontView);

    pControls=uipanel('Parent',fig,'Title','3D display and playback', ...
        'Units','normalized','Position',[.023 .021 .445 .252], ...
        'BackgroundColor',panelBg,'ForegroundColor',fg,'FontWeight','bold');
    cbCloud=makeCheck(pControls,'Point cloud', [.02 .76 .19 .18],1);
    cbLM=makeCheck(pControls,'Landmarks', [.23 .76 .19 .18],1);
    cbLines=makeCheck(pControls,'Show A-B distance line', [.44 .76 .25 .18],0);
    cbText=makeCheck(pControls,'Landmark labels', [.71 .76 .28 .18],0);
    % Keep the legacy all-landmarks checkbox in solid mode, and add three
    % explicit 3D display styles. RGB/depth modes force the full landmark overlay.
    cbAllLM=makeCheck(pControls,'All 21 (solid)',[.02 .54 .195 .19],0);
    cbDenseLM=makeCheck(pControls,'Dense eyes/brows', [.22 .54 .265 .19],1);
    set(cbDenseLM,'TooltipString', ...
        'Additional MediaPipe eye, eyelid, eyebrow and nose mesh vertices where recorded 3D depth exists.');
    cloudModeMenu=uicontrol(pControls,'Style','popupmenu','Units','normalized', ...
        'Position',[.49 .54 .49 .19], ...
        'String',{'Solid cloud (current)'; ...
        'Recorded RGB + all landmark dots'; ...
        'Depth + all landmark dots'}, ...
        'Value',1,'BackgroundColor','w','Callback',@refresh, ...
        'Tag','ThreeCloudDisplayModes', ...
        'TooltipString',['Solid: current single-color cloud; RGB: original same-frame RGB; ' ...
        'Depth: same-frame camera Z in mm. RGB/depth show all available landmark XYZ.']);
    cbAvg=makeCheck(pControls,'Moving average', [.02 .43 .30 .13],0);
    uicontrol(pControls,'Style','text','Units','normalized','Position',[.35 .43 .16 .12], ...
        'String','N frames:','ForegroundColor',fg,'BackgroundColor',panelBg);
    avgEdit=uicontrol(pControls,'Style','edit','Units','normalized', ...
        'Position',[.50 .425 .19 .15],'String',num2str(movingWindow), ...
        'Callback',@changeAverage,'BackgroundColor','w', ...
        'Tag','NFrameMovingAverageWindow', ...
        'TooltipString',['Enter a positive integer N and press Enter. ' ...
        'The trailing mean uses the current frame and up to N-1 previous frames; ' ...
        'N may exceed frames available so far. Missing values are ignored.']);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.71 .44 .27 .16],'String','Enlarged 3D', ...
        'Callback',@openEnlarged3D);
    cbSmooth=makeCheck(pControls,'Median landmark smoothing',[.02 .315 .34 .105],0);
    set(cbSmooth,'Callback',@changeSmoothing);
    cbDotLabels=makeCheck(pControls,'Label graph dots',[.35 .315 .31 .105],0);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.71 .30 .27 .13],'String','ASI / Measure', ...
        'Callback',@toggleViewMode);
    slider=uicontrol(pControls,'Style','slider','Units','normalized', ...
        'Position',[.02 .258 .65 .06],'Min',1,'Max',max(2,nFrames),'Value',max(1,nFrames), ...
        'SliderStep',[1/max(1,nFrames-1) min(1,5/max(1,nFrames-1))], 'Callback',@slideFrame);
    frameText=uicontrol(pControls,'Style','text','Units','normalized', ...
        'Position',[.70 .205 .28 .09],'String','','ForegroundColor',fg, ...
        'BackgroundColor',panelBg,'HorizontalAlignment','left');
    startButton=uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.02 .035 .17 .155],'String','Play replay','Callback',@togglePlay);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.20 .035 .14 .155],'String','Pair / formula table','Callback',@showDetails);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.35 .035 .13 .155],'String','Export tables','Callback',@exportTables);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.49 .035 .14 .155],'String','Choose landmarks', ...
        'Callback',@chooseLandmarks);
    uicontrol(pControls,'Style','pushbutton','Units','normalized', ...
        'Position',[.64 .035 .16 .155],'String','L/R review', ...
        'Callback',@openReview);
    uicontrol(pControls,'Style','text','Units','normalized', ...
        'Position',[.805 .06 .055 .11],'String','FPS:','ForegroundColor',fg, ...
        'BackgroundColor',panelBg,'HorizontalAlignment','left');
    fpsString='';if isfinite(fps),fpsString=sprintf('%.3g',fps);end
    fpsEdit=uicontrol(pControls,'Style','edit','Units','normalized', ...
        'Position',[.86 .055 .12 .13],'String',fpsString,'BackgroundColor',[1 1 1], ...
        'TooltipString','Original camera acquisition FPS, not playback speed. Leave blank if unknown.','Callback',@changeFPS);

    panelU=createASIBox('UPPER FACE','Selected upper landmarks',.730,upperColor);
    panelL=createASIBox('LOWER FACE','Selected lower landmarks',.491,lowerColor);
    panelO=createASIBox('OVERALL FACE','Pooled selected features',.252,overallColor);

    % Eyelid investigation opens an independent frame-linked plot.
    uicontrol(fig,'Style','pushbutton','Units','normalized', ...
        'Position',[.870 .955 .113 .035],'String','Eyelid / blink QC', ...
        'TooltipString','Experimental upper-to-lower eyelid opening in mm. Requires FaceASI v52 dense vertices.', ...
        'Callback',@openEyelidQC);

    % Expanded right-hand DISTANCE review (shown instead of ASI summary cards).
    measurePanel=uipanel('Parent',fig,'Title','3D MEASUREMENT | Select any two recorded landmarks', ...
        'Units','normalized','Position',[.490 .250 .497 .695], ...
        'BackgroundColor',panelBg,'ForegroundColor',fg,'FontWeight','bold');
    uicontrol(measurePanel,'Style','text','Units','normalized', ...
        'Position',[.025 .912 .14 .055],'String','Landmark A', ...
        'BackgroundColor',panelBg,'ForegroundColor',selectionColorA, ...
        'HorizontalAlignment','left','FontWeight','bold');
    uicontrol(measurePanel,'Style','text','Units','normalized', ...
        'Position',[.52 .912 .14 .055],'String','Landmark B', ...
        'BackgroundColor',panelBg,'ForegroundColor',selectionColorB, ...
        'HorizontalAlignment','left','FontWeight','bold');
    allChoices=cell(nLM,1);
    for z=1:nLM,allChoices{z}=sprintf('%s  [%s]',landmarkShortNames{z},IDs{z});end
    menuA=uicontrol(measurePanel,'Style','popupmenu','Units','normalized', ...
        'Position',[.025 .845 .455 .060],'String',allChoices,'Value',selectionA, ...
        'BackgroundColor','w','Callback',@changeSelection, ...
        'Tag','SelectedLandmarkA');
    menuB=uicontrol(measurePanel,'Style','popupmenu','Units','normalized', ...
        'Position',[.52 .845 .455 .060],'String',allChoices,'Value',selectionB, ...
        'BackgroundColor','w','Callback',@changeSelection, ...
        'Tag','SelectedLandmarkB');
    measureNote=uicontrol(measurePanel,'Style','text','Units','normalized', ...
        'Position',[.030 .700 .94 .131],'String','', ...
        'BackgroundColor',panelBg,'ForegroundColor',fg,'FontSize',9, ...
        'HorizontalAlignment','left','FontName','Consolas');
    measureAx=axes('Parent',measurePanel,'Units','normalized', ...
        'Position',[.08 .208 .89 .480], ...
        'Color',[.085 .11 .16],'XColor',fg,'YColor',fg);
    hold(measureAx,'on');grid(measureAx,'on');
    measureTrace=plot(measureAx,nan,nan,'-','Color',selectionColorA, ...
        'LineWidth',2,'DisplayName','Actual A-to-B 3D distance');
    measureDots=plot(measureAx,nan,nan,'o','Color',selectionColorB, ...
        'MarkerFaceColor',selectionColorB,'MarkerSize',5,'LineStyle','none', ...
        'DisplayName','Selected frame measurements');
    measureAvg=plot(measureAx,nan,nan,'--','Color',[.26 .88 .73], ...
        'LineWidth',2,'DisplayName','Trailing moving average');
    measureCursor=plot(measureAx,nan,nan,'-','Color',[1 .95 .6], ...
        'LineWidth',1.2,'HandleVisibility','off');
    xlabel(measureAx,'Frame');ylabel(measureAx,'Euclidean distance (mm)');
    legend(measureAx,'Location','best','TextColor',fg,'FontSize',8);
    measureStats=uicontrol(measurePanel,'Style','text','Units','normalized', ...
        'Position',[.035 .077 .70 .125],'String','', ...
        'BackgroundColor',panelBg,'ForegroundColor',fg, ...
        'HorizontalAlignment','left','FontSize',9,'FontName','Consolas');
    uicontrol(measurePanel,'Style','pushbutton','Units','normalized', ...
        'Position',[.753 .100 .22 .080],'String','Enlarge + all values', ...
        'Callback',@openMeasurementGraph);
    uicontrol(measurePanel,'Style','pushbutton','Units','normalized', ...
        'Position',[.753 .015 .22 .075],'String','Inspect dots', ...
        'Callback',@toggleDotInspection);
    pv=uipanel('Parent',fig,'Title','Frame-to-frame 3D landmark movement', ...
        'Units','normalized','Position',[.490 .022 .497 .210], ...
        'BackgroundColor',panelBg,'ForegroundColor',fg,'FontWeight','bold');
    velTxt=uicontrol(pv,'Style','text','Units','normalized', ...
        'Position',[.025 .13 .62 .77],'BackgroundColor',panelBg,'ForegroundColor',fg, ...
        'FontSize',9,'HorizontalAlignment','left','String','');
    velAx=axes('Parent',pv,'Units','normalized','Position',[.70 .25 .27 .55], ...
        'Color',[.08 .11 .15],'XColor',fg,'YColor',fg);hold(velAx,'on');grid(velAx,'on');
    velTrace=plot(velAx,nan,nan,'-','Color',lowerColor,'LineWidth',1.7);
    velCursor=plot(velAx,nan,nan,'-','Color',[1 .45 .35]);
    xlabel(velAx,'Frame');

    set(fig,'CloseRequestFcn',@closeViewer);
    % The timer only advances prerecorded frames or calls user-provided
    % acquisition. It does not impose source timestamps or camera FPS.
    clockTimer=timer('ExecutionMode','fixedSpacing','Period',opt.UpdatePeriod_s, ...
        'BusyMode','drop','TimerFcn',@timerTick);
    if hasProvider
        set(startButton,'String','Start LIVE feed');
    else
        set(startButton,'String',sprintf('Play %d frames',nFrames));
        fitCloud();
    end
    if nFrames>0,frame=1;end
    enableRotate();
    refresh();
    Results=makeResults();
    if hasCSV
        setappdata(fig,'DeviceScan',S); % exact original scores, mesh and 2D pixels
        showDeviceResults();
    end
    if nargout==0,assignin('base','FacialASI_Results',Results);end

    function indexes=idList(names)
        indexes=zeros(1,numel(names));
        for t=1:numel(names)
            m=find(strcmp(IDs,names{t}),1);
            if isempty(m),error('Missing required landmark ID: %s',names{t});end
            indexes(t)=m;
        end
    end

    function c=makeCheck(parent,string,pos,value)
        c=uicontrol(parent,'Style','checkbox','Units','normalized', ...
            'Position',pos,'String',string,'Value',value,'Callback',@refresh, ...
            'BackgroundColor',panelBg,'ForegroundColor',fg);
    end

    function b=createASIBox(header,subtitle,y,c)
        b.p=uipanel('Parent',fig,'Title',[header ' | ' subtitle], ...
            'Units','normalized','Position',[.490 y .497 .215], ...
            'BackgroundColor',panelBg,'ForegroundColor',c,'FontWeight','bold');
        % Each name represents ONE actual anatomical landmark, not a pair.
        % Midface landmarks appear in LOWER for consistency with existing ASI.
        grp=1;
        if strcmp(header,'LOWER FACE'),grp=2;
        elseif strcmp(header,'OVERALL FACE'),grp=3;end
        if grp==1
            b.graphLM=unique([pairL(1:3) pairR(1:3) upperDisplayIDs(:)' idNasion],'stable');
            b.graphPairIndices=upperIdx;
        elseif grp==2
            b.graphLM=unique([pairL(5) pairR(5) pairL(4) pairR(4) ...
                pairL(6) pairR(6) lowerDisplayIDs(:)' ...
                setdiff(midDisplayIDs,idNasion,'stable')],'stable');
            b.graphPairIndices=lowerIdx;
        else
            b.graphLM=unique([pairL(5) pairR(5) 1:nLM],'stable');
            b.graphPairIndices=1:6;
        end
        names=cell(numel(b.graphLM),1);
        for j=1:numel(b.graphLM)
            id=b.graphLM(j);
            names{j}=sprintf('%s [%s]',landmarkShortNames{id},IDs{id});
        end
        % Aggregates and bilateral diagnostic values stay AVAILABLE, but
        % are clearly separated from individual landmark observations.
        names=[names;{'ASI TOTAL (pooled regional, not one landmark)'}];
        for j=b.graphPairIndices
            names{end+1,1}=sprintf('PAIR ASI - %s LEFT/RIGHT',pairNames{j});
        end
        b.graphCaption=uicontrol(b.p,'Style','text','Units','normalized', ...
            'Position',[.71 .926 .28 .055],'String','Individual landmark:', ...
            'BackgroundColor',panelBg,'ForegroundColor',fg, ...
            'HorizontalAlignment','left','FontSize',8);
        b.graphMenu=uicontrol(b.p,'Style','popupmenu','Units','normalized', ...
            'Position',[.68 .795 .31 .120],'String',names,'Value',1, ...
            'Callback',@refresh,'BackgroundColor','w', ...
            'Tag',sprintf('ASIIndividualGraphMenu%d',grp), ...
            'TooltipString','Choose ONE landmark. Aggregate ASIs are listed separately.');
        b.metricMenu=uicontrol(b.p,'Style','popupmenu','Units','normalized', ...
            'Position',[.68 .674 .31 .105],'String',landmarkMetricOptions, ...
            'Value',1,'Callback',@refresh,'BackgroundColor','w', ...
            'Tag',sprintf('ASIIndividualMetricMenu%d',grp), ...
            'TooltipString','How to plot the selected individual 3D landmark');
        b.t=uicontrol(b.p,'Style','text','Units','normalized', ...
            'Position',[.025 .035 .655 .94],'BackgroundColor',panelBg, ...
            'ForegroundColor',fg,'HorizontalAlignment','left','FontName','Consolas', ...
            'FontSize',8,'String','');
        b.a=axes('Parent',b.p,'Units','normalized','Position',[.69 .26 .30 .43], ...
            'Color',[.08 .11 .15],'XColor',fg,'YColor',fg);
        hold(b.a,'on');grid(b.a,'on');
        b.trace=plot(b.a,nan,nan,'-o','Color',c,'LineWidth',1.35, ...
            'MarkerSize',3,'MarkerFaceColor',c,'HitTest','on');
        b.averageTrace=plot(b.a,nan,nan,'--','Color',[.30 .87 .70], ...
            'LineWidth',1.6,'Visible','off');
        b.cursor=plot(b.a,nan,nan,'-','Color',[1 .45 .38], ...
            'HitTest','off','PickableParts','none');
        setappdata(b.p,'DotLabels',gobjects(0,1));
        b.current=uicontrol(b.p,'Style','text','Units','normalized', ...
            'Position',[.700 .185 .292 .112],'String','', ...
            'BackgroundColor',panelBg,'ForegroundColor',fg, ...
            'HorizontalAlignment','left','FontSize',8);
        b.tableButton=uicontrol(b.p,'Style','pushbutton','Units','normalized', ...
            'Position',[.70 .018 .29 .152],'String','All 30 dot values / larger plot', ...
            'FontSize',8,'Callback',@(~,~)openIndividualGraphTable(grp));
        xlabel(b.a,'Frame');ylabel(b.a,'Distance (mm)');
    end

    function m=emptyFrame()
        p=struct('label','','L','','R','','valid',false, ...
            'hL',NaN,'hR',NaN,'width',NaN,'dH',NaN,'dV',NaN, ...
            'dD',NaN,'Lxyz',nan(1,3),'Rxyz',nan(1,3), ...
            'midxyz',nan(1,3),'footLxyz',nan(1,3),'footRxyz',nan(1,3), ...
            'lvpoint',nan(1,3),'ldpoint',nan(1,3), ...
            'Luvw',nan(1,3),'Ruvw',nan(1,3),'miduvw',nan(1,3));
        z=struct('H',NaN,'V',NaN,'D',NaN,'ASI',NaN, ...
            'sumH',NaN,'sumV',NaN,'sumD',NaN,'sumW',NaN, ...
            'valid',false,'count',0,'needed',0,'possible',0, ...
            'bilateralASI',NaN,'M',0,'sumM',0,'midCount',0,'midPossible',0);
        m=struct('pairs',repmat(p,6,1),'upper',z,'lower',z,'overall',z, ...
            'midlineXYZ',nan(0,3),'origin',nan(1,3), ...
            'basis',nan(3,3),'validLandmarks',0);
    end

    function [p,q,m]=processFrame(c,seeds)
        p=nan(nLM,3);q=nan(nLM,3);m=emptyFrame();
        if hasCSV
            p=double(seeds);[q,m]=geometryOnly(p);return; % Never snap measured labels to a thinned cloud.
        end
        if isempty(c)||size(c,2)<3||size(c,1)<3||size(seeds,1)~=nLM
            return;
        end
        good=all(isfinite(c(:,1:3)),2);pts=double(c(good,1:3));
        if size(pts,1)<3,return;end
        for ii=1:nLM
            seed=double(seeds(ii,:));
            if ~all(isfinite(seed)),continue;end
            ds=bsxfun(@minus,pts,seed);
            [v,at]=min(sum(ds.^2,2));
            if v<=snapRadius^2,p(ii,:)=pts(at,:);end
        end
        [q,m]=geometryOnly(p);
    end

    function r=regional(p,q,idx,chosenMid)
        r=struct('H',NaN,'V',NaN,'D',NaN,'ASI',NaN, ...
            'sumH',NaN,'sumV',NaN,'sumD',NaN,'sumW',NaN, ...
            'valid',false,'count',0,'needed',1,'possible',numel(idx), ...
            'bilateralASI',NaN,'M',NaN,'sumM',NaN, ...
            'midCount',0,'midPossible',numel(chosenMid));
        if isempty(idx),return;end
        valid=false(1,numel(idx));
        for z=1:numel(idx),valid(z)=p(idx(z)).valid;end
        r.count=sum(valid);
        % Require at least one selected VALID bilateral pair in a region.
        % In the Overall panel require BOTH selected upper + lower data.
        if r.count<1,return;end
        pp=p(idx(valid));
        r.sumH=sum([pp.dH]);r.sumV=sum([pp.dV]);
        r.sumD=sum([pp.dD]);r.sumW=sum([pp.width]);
        if r.sumW<=0||~isfinite(r.sumW),return;end
        r.H=100*r.sumH/r.sumW;r.V=100*r.sumV/r.sumW;
        r.D=100*r.sumD/r.sumW;
        r.bilateralASI=sqrt(r.H^2+r.V^2+r.D^2);
        r.sumM=0;
        for z=chosenMid(:)'
            if ~all(isfinite(q(z,:))),continue;end
            % The anchors define the axis and offer no independent offset.
            if z==idNasion||z==idSubnasale,continue;end
            % Reject gross off-face depth artifacts (e.g. MENTON in capture).
            if abs(q(z,3))>65,continue;end
            r.sumM=r.sumM+2*abs(q(z,1));
            r.midCount=r.midCount+1;
        end
        r.M=100*r.sumM/r.sumW;
        r.ASI=sqrt(r.H^2+r.V^2+r.D^2+r.M^2);
        r.valid=true;
    end

    function recomputeAll()
        for f=1:nFrames
            q=squeeze(localXYZ(f,:,:));
            m=measurements(f);
            u=upperIdx(selectedPairs(upperIdx));
            l=lowerIdx(selectedPairs(lowerIdx));
            ui=upperMid(selectedMid(upperMid));
            li=lowerMid(selectedMid(lowerMid));
            m.upper=regional(m.pairs,q,u,ui);
            m.lower=regional(m.pairs,q,l,li);
            m.overall=regional(m.pairs,q,[u l],[ui li]);
            if ~m.upper.valid||~m.lower.valid
                m.overall.valid=false;
                m.overall.ASI=NaN;
                m.overall.H=NaN;m.overall.V=NaN;
                m.overall.D=NaN;m.overall.M=NaN;
            end
            measurements(f)=m;
        end
    end

    function P=causalSmooth(P)
        % Robust 3-frame CAUSAL coordinate median; no future frames.
        % Raw labels retained separately and are never silently modified.
        for f=2:size(P,1)
            lo=max(1,f-2);
            for i=1:nLM
                a=squeeze(rawProcessed(lo:f,i,:));
                if size(a,1)==1,a=reshape(a,1,3);end
                if ~all(isfinite(squeeze(rawProcessed(f,i,:)))),continue;end
                for d=1:3
                    vv=a(:,d);vv=vv(isfinite(vv));
                    if numel(vv)>=2,P(f,i,d)=median(vv);end
                end
            end
        end
    end

    function recalcWithMode()
        if smoothMode, seeds=smoothProcessed;else,seeds=rawProcessed;end
        for f=1:nFrames
            % Use smooth XYZ in q geometry; do not nearest-snap again.
            [q,m]=geometryOnly(squeeze(seeds(f,:,:)));
            processed(f,:,:)=seeds(f,:,:);
            localXYZ(f,:,:)=reshape(q,1,nLM,3);
            measurements(f)=m;
        end
        recomputeAll();
    end

    function [q,m]=geometryOnly(p)
        % Calculate geometry of already cloud-snapped/smoothed points.
        q=nan(nLM,3);m=emptyFrame();
        for t=1:6
            m.pairs(t).label=pairNames{t};
            m.pairs(t).L=IDs{pairL(t)};m.pairs(t).R=IDs{pairR(t)};
        end
        good=all(isfinite(p),2);m.validLandmarks=sum(good);
        if ~good(idNasion)||~good(idSubnasale),return;end
        origin=p(idSubnasale,:);
        v=p(idSubnasale,:)-p(idNasion,:);
        if norm(v)<5,return;end
        v=v/norm(v);
        lr=nan(5,3);
        for j=1:5
            if good(pairL(j))&&good(pairR(j))
                lr(j,:)=p(pairL(j),:)-p(pairR(j),:);
                lr(j,:)=lr(j,:)/max(norm(lr(j,:)),eps);
            end
        end
        valid=all(isfinite(lr),2);
        if sum(valid)<2,return;end
        u=mean(lr(valid,:),1);u=u-dot(u,v)*v;
        if norm(u)<.2,return;end
        u=u/norm(u);w=cross(u,v);w=w/norm(w);
        basis=[u(:),v(:),w(:)];
        q=bsxfun(@minus,p,origin)*basis;
        m.origin=origin;m.basis=basis;
        along=linspace(-80,110,11)';
        m.midlineXYZ=bsxfun(@plus,origin,along*basis(:,2)');
        for j=1:6
            t=m.pairs(j);L=pairL(j);R=pairR(j);
            if ~good(L)||~good(R),continue;end
            hl=abs(q(L,1));hr=abs(q(R,1));wd=hl+hr;
            if wd<8||q(L,1)<=0||q(R,1)>=0,continue;end
            % No big off-face depth points, even if they have a label.
            if abs(q(L,3))>65||abs(q(R,3))>65,continue;end
            meanV=(q(L,2)+q(R,2))/2;
            t.valid=true;t.hL=hl;t.hR=hr;t.width=wd;
            t.dH=abs(hl-hr);t.dV=abs(q(L,2)-q(R,2));
            t.dD=abs(q(L,3)-q(R,3));
            t.Lxyz=p(L,:);t.Rxyz=p(R,:);
            t.miduvw=[0 meanV 0];t.Luvw=q(L,:);t.Ruvw=q(R,:);
            t.midxyz=origin+[0 meanV 0]*basis';
            % Perpendicular feet on U=0 are the true hL/hR line endpoints.
            t.footLxyz=origin+[0 q(L,2) q(L,3)]*basis';
            t.footRxyz=origin+[0 q(R,2) q(R,3)]*basis';
            t.lvpoint=origin+[q(L,1) q(R,2) q(L,3)]*basis';
            t.ldpoint=origin+[q(L,1) q(R,2) q(R,3)]*basis';
            m.pairs(j)=t;
        end
    end

    function [disp3,change,description]=movementSeries()
        % Per-landmark 3D velocity vectors in local face coordinates.
        % 'change' is RMS of each landmark's delta-velocity vector,
        % not the difference between nonnegative speeds.
        disp3=nan(nFrames,1);change=nan(nFrames,1);
        description='mm/frame';
        useTiming=false;
        if isfinite(fps),description='mm/s';useTiming=true;
        elseif nFrames>=2&&all(isfinite(acquiredTime)) ...
                && all(diff(acquiredTime)>0)
            description='mm/s';useTiming=true;
        end
        vec=nan(nFrames,numel(trackIDs),3);
        dt=nan(nFrames,1);
        for ff=2:nFrames
            a=squeeze(localXYZ(ff-1,trackIDs,:));
            b=squeeze(localXYZ(ff,trackIDs,:));
            both=all(isfinite(a),2)&all(isfinite(b),2);
            if sum(both)<4,continue;end
            interval=1;
            if useTiming
                if all(isfinite(acquiredTime(ff-1:ff))) ...
                        && acquiredTime(ff)>acquiredTime(ff-1)
                    interval=acquiredTime(ff)-acquiredTime(ff-1);
                elseif isfinite(fps),interval=1/fps;
                else,continue;end
            end
            dt(ff)=interval;
            vv=(b(both,:)-a(both,:))/interval;
            vec(ff,both,:)=reshape(vv,[1 sum(both) 3]);
            disp3(ff)=sqrt(mean(sum(vv.^2,2)));
        end
        for ff=3:nFrames
            va=squeeze(vec(ff-1,:,:));vb=squeeze(vec(ff,:,:));
            both=all(isfinite(va),2)&all(isfinite(vb),2);
            if sum(both)<4||~isfinite(dt(ff)),continue;end
            deltaVec=(vb(both,:)-va(both,:))/dt(ff);
            change(ff)=sqrt(mean(sum(deltaVec.^2,2)));
        end
    end

    function refresh(varargin)
        if ~isgraphics(fig),return;end
        if nFrames==0
            set(frameText,'String','No live frames yet');
            set(velTxt,'String',sprintf('Select Start LIVE feed.\nAwaiting external data source.'));
            return;
        end
        frame=max(1,min(frame,nFrames)); c=rawClouds{frame};
        p=squeeze(processed(frame,:,:));m=measurements(frame);
        cloudStyle=get(cloudModeMenu,'Value');
        % All modes use the SAME recorded XYZ array and frame indices.
        % No smoothing/recalculation is introduced by changing color style.
        [xyz,pointColors,depthLimits,cloudDescription]=cloudColors(c,cloudStyle);
        if get(cbCloud,'Value') && ~isempty(xyz)
            set(cloudH,'XData',xyz(:,1),'YData',xyz(:,2), ...
                'ZData',xyz(:,3),'CData',pointColors,'Visible','on');
            if ~isempty(depthLimits)
                colormap(ax,parula(256));caxis(ax,depthLimits);
            end
        else
            set(cloudH,'Visible','off');
        end
        set(cloudH,'UserData',struct('kind','Cloud point'));
        lmOn=get(cbLM,'Value')~=0;
        % The checkbox retains its old meaning in SOLID mode.
        % RGB and depth modes automatically show all measured landmark dots.
        allMode=(cloudStyle~=1) || (get(cbAllLM,'Value')~=0);
        set(cbAllLM,'Enable',onoff(cloudStyle==1));
        if cloudStyle==1
            set([marksU marksL marksM marksOther],'CData',[.70 .77 .81]);
        else
            % Anatomical overlay colors (separate from the cloud colors).
            set(marksU,'CData',upperColor);
            set(marksM,'CData',midColor);
            set(marksL,'CData',lowerColor);
            set(marksOther,'CData',[1.00 .89 .39]);
        end
        if allMode
            assignMarks(marksU,upperDisplayIDs,lmOn);
            assignMarks(marksM,midDisplayIDs,lmOn);
            assignMarks(marksL,lowerDisplayIDs,lmOn);
            used=unique([upperDisplayIDs midDisplayIDs lowerDisplayIDs]);
            assignMarks(marksOther,setdiff(1:nLM,used),lmOn);
        else
            set([marksU marksM marksL marksOther],'Visible','off');
        end
        % Legacy bilateral midpoint/projection geometry is NOT plotted:
        % those lines were unrelated to the chosen A-to-B graph.
        set([midLine;midMeasLines(:);measLines(:)],'Visible','off');
        set([midMeasText(:);measText(:)],'Visible','off');
        for j=1:nLM
            showLabel=get(cbText,'Value') && lmOn && ...
                (allMode || j==selectionA || j==selectionB) && ...
                all(isfinite(p(j,:)));
            if showLabel
                set(lmLabels(j),'Position',p(j,:)+[1.2 1.2 0], ...
                    'String',IDs{j},'Color',fg,'Visible','on');
            else,set(lmLabels(j),'Visible','off');end
        end
        % Dense mesh layers use their OWN recorded 3D positions (not an
        % interpolation from 21 named points or the RGB cloud color).
        denseAvailable=any(isfinite(denseXYZ(:)));
        set(cbDenseLM,'Enable',onoff(denseAvailable));
        if ~denseAvailable
            set(cbDenseLM,'String','Dense XYZ unavailable');
        else
            set(cbDenseLM,'String','Dense eyes/brows');
        end
        denseOn=get(cbDenseLM,'Value')~=0 && denseAvailable && lmOn;
        drawDense(denseEyeH,denseEyeIDs,denseOn);
        drawDense(denseBrowH,denseBrowIDs,denseOn);
        drawDense(denseNoseH,denseNoseIDs,denseOn);
        renderSelectedPair(p);
        upper=nan(nFrames,1);lower=upper;overall=upper;
        for ff=1:nFrames
            upper(ff)=measurements(ff).upper.ASI;
            lower(ff)=measurements(ff).lower.ASI;
            overall(ff)=measurements(ff).overall.ASI;
        end
        renderIndex(panelU,upper,m.upper,'UPPER',upperIdx(selectedPairs(upperIdx)));
        renderIndex(panelL,lower,m.lower,'LOWER',lowerIdx(selectedPairs(lowerIdx)));
        renderIndex(panelO,overall,m.overall,'OVERALL',find(selectedPairs));
        showMeasurement=strcmp(viewMode,'measure');
        set(measurePanel,'Visible',onoff(showMeasurement));
        for one=[panelU.p panelL.p panelO.p]
            set(one,'Visible',onoff(~showMeasurement));
        end
        set([menuA menuB cbLines],'Enable',onoff(showMeasurement));
        renderMeasurementGraph();
        [step,delta,unit]=movementSeries();
        set(velTrace,'XData',frameNumbers,'YData',step);
        set(velAx,'YLim',smartLimit(step),'XLim',smartX());
        set(velCursor,'XData',[frameNumbers(frame) frameNumbers(frame)],'YData',ylim(velAx));
        ylabel(velAx,unit);
        if strcmp(unit,'mm/s'),aunit='mm/s^2';else,aunit='mm/frame^2';end
        good=step(1:frame);good=good(isfinite(good));
        if isempty(good),avg='N/A';else,avg=sprintf('%.2f',mean(good));end
        if numel(good)<2,sdev='N/A';else,sdev=sprintf('%.2f',std(good,0));end
        set(velTxt,'String',sprintf([ ...
            'RMS face-local landmark speed: %s\n' ...
            'Running speed mean: %s | SD: %s\n' ...
            'RMS velocity-change rate: %s\n' ...
            'Fixed 10 tracked landmarks; >=4 needed\n' ...
            'Unit: %s | Change unit: %s\n' ...
            'Face basis changes between frames'], ...
            formatted(step(frame)),avg,sdev,formatted(delta(frame)),unit,aunit));
        set(frameText,'String',sprintf('Frame %d/%d | %d/%d valid XYZ', ...
            frame,nFrames,m.validLandmarks,nLM));
        set(slider,'Min',1,'Max',max(2,nFrames),'Value',frame, ...
            'SliderStep',[1/max(1,nFrames-1) min(1,5/max(1,nFrames-1))]);
        title(ax,sprintf('Frame %d | %s | %d/%d available 3D landmarks', ...
            frame,cloudDescription,m.validLandmarks,nLM), ...
            'Color',fg,'FontSize',10,'Interpreter','none');
        updateReview();
        updateEyelidQC();
        drawnow limitrate;
    end

    function drawDense(handle,requestedIDs,shown)
        if ~shown || frame<1 || frame>size(denseXYZ,1)
            set(handle,'Visible','off');return;
        end
        [tf,index]=ismember(requestedIDs,denseIDs);
        actualIDs=requestedIDs(tf);index=index(tf);
        P=squeeze(denseXYZ(frame,index,:));
        if isempty(P),set(handle,'Visible','off');return;end
        P=reshape(P,[],3);
        valid=all(isfinite(P),2);P=P(valid,:);actualIDs=actualIDs(valid);
        if isempty(P),set(handle,'Visible','off');return;end
        set(handle,'XData',P(:,1),'YData',P(:,2),'ZData',P(:,3), ...
            'UserData',struct('kind','Dense mesh vertex','meshIDs',actualIDs), ...
            'Visible','on');
    end

    function [apL,apR]=eyelidSeries()
        apL=nan(nFrames,1);apR=nan(nFrames,1);
        ilU=find(denseIDs==eyelidLeftUpper,1);
        ilD=find(denseIDs==eyelidLeftLower,1);
        irU=find(denseIDs==eyelidRightUpper,1);
        irD=find(denseIDs==eyelidRightLower,1);
        if any(cellfun(@isempty,{ilU,ilD,irU,irD})),return;end
        for ff=1:nFrames
            m=measurements(ff);
            if ~all(isfinite(m.basis(:))),continue;end
            axisV=m.basis(:,2);
            li=squeeze(denseXYZ(ff,[ilU ilD],:));
            ri=squeeze(denseXYZ(ff,[irU irD],:));
            if all(isfinite(li(:)))
                apL(ff)=abs(dot(li(1,:)-li(2,:),axisV'));
            end
            if all(isfinite(ri(:)))
                apR(ff)=abs(dot(ri(1,:)-ri(2,:),axisV'));
            end
        end
    end

    function openEyelidQC(~,~)
        if ~any(isfinite(denseXYZ(:)))
            warndlg(['The old embedded recording includes only 21 labeled ' ...
                'XYZ points; it does NOT contain dense eyelid/eyebrow vertices. ' ...
                'Capture with FaceASI v52 and import the burst with the ' ...
                'dense-aware importer. No eyelid coordinates are fabricated.'], ...
                'Dense mesh data unavailable');return;
        end
        if ~isempty(eyelidFig)&&isgraphics(eyelidFig)
            figure(eyelidFig);updateEyelidQC();return;
        end
        eyelidLines=gobjects(1,6);
        eyelidFig=figure('Name','Experimental eyelid opening / blink QC (not clinical ptosis)', ...
            'NumberTitle','off','Color','w','Position',[90 90 1250 660]);
        a=axes('Parent',eyelidFig,'Units','normalized', ...
            'Position',[.08 .30 .88 .62]);hold(a,'on');grid(a,'on');
        eyelidLines(1)=plot(a,nan,nan,'-o','Color',[.20 .46 .92], ...
            'MarkerSize',4,'DisplayName','Left raw vertical aperture');
        eyelidLines(2)=plot(a,nan,nan,'-o','Color',[.94 .31 .23], ...
            'MarkerSize',4,'DisplayName','Right raw vertical aperture');
        eyelidLines(3)=plot(a,nan,nan,'--','Color',[.15 .40 .70], ...
            'LineWidth',2,'DisplayName','Left trailing N mean');
        eyelidLines(4)=plot(a,nan,nan,'--','Color',[.73 .19 .16], ...
            'LineWidth',2,'DisplayName','Right trailing N mean');
        eyelidLines(5)=plot(a,nan,nan,':','Color',[.42 .38 .46], ...
            'DisplayName','Absolute left-right aperture difference');
        eyelidLines(6)=plot(a,nan,nan,'-','Color',[.7 .12 .70], ...
            'HandleVisibility','off');
        xlabel(a,'Frame');ylabel(a,'Vertical eyelid-opening proxy (mm)');
        legend(a,'Location','best');
        eyelidInfo=uicontrol(eyelidFig,'Style','text','Units','normalized', ...
            'Position',[.08 .035 .88 .19],'HorizontalAlignment','left', ...
            'BackgroundColor','w','FontSize',10,'String','');
        updateEyelidQC();
    end

    function updateEyelidQC()
        if isempty(eyelidFig)||~isgraphics(eyelidFig) || numel(eyelidLines)~=6
            return;
        end
        [L,R]=eyelidSeries();
        La=trailingAverage(L,movingWindow);Ra=trailingAverage(R,movingWindow);
        values={L,R,La,Ra,abs(L-R)};
        for ii=1:5
            set(eyelidLines(ii),'XData',frameNumbers,'YData',values{ii});
        end
        set(eyelidLines(3:4),'Visible',onoff(get(cbAvg,'Value')~=0));
        a=ancestor(eyelidLines(1),'axes');
        if isgraphics(a)
            lim=smartLimitSigned([L;R;La;Ra;abs(L-R)]);
            xlim(a,smartX());ylim(a,lim);
            set(eyelidLines(6),'XData',[frameNumbers(frame) frameNumbers(frame)], ...
                'YData',lim);
        end
        set(eyelidInfo,'String',sprintf([ ...
            'Frame %d: left %s mm, right %s mm; absolute mismatch %s mm\n' ...
            'N=%d trailing mean: left %s mm; right %s mm  |  left SD %s, right SD %s mm\n' ...
            'Measured from mesh 386/374 and 159/145 along the estimated face V-axis. ' ...
            'This is an eyelid opening proxy, NOT a diagnosis of ptosis/MRD1.'], ...
            frameNumbers(frame),formatted(L(frame)),formatted(R(frame)), ...
            formatted(abs(L(frame)-R(frame))),movingWindow, ...
            formatted(La(frame)),formatted(Ra(frame)), ...
            formatted(safeSD(L)),formatted(safeSD(R))));
    end

    function [xyz,colors,depthLims,description]=cloudColors(data,style)
        % Display-only mapping; never modifies captured XYZ/RGB or ASI.
        xyz=zeros(0,3);colors=zeros(0,3);depthLims=[];
        switch style
            case 2,description='Recorded RGB';
            case 3,description='Depth (camera Z)';
            otherwise,description='Solid cloud';
        end
        if isempty(data)||~isnumeric(data)||size(data,2)<3,return;end
        finiteXYZ=all(isfinite(data(:,1:3)),2);
        xyz=double(data(finiteXYZ,1:3));
        if isempty(xyz),return;end
        switch style
            case 2
                if size(data,2)>=6
                    colors=double(data(finiteXYZ,4:6));
                    bad=~all(isfinite(colors),2);
                    if any(isfinite(colors(:)) & colors(:)>1)
                        % Embedded recorded RGB is stored in 0...255.
                        colors=colors/255;
                    end
                    colors=max(0,min(1,colors));
                    if any(bad)
                        colors(bad,:)=repmat([.43 .60 .67],sum(bad),1);
                    end
                else
                    % External Nx3 providers contain no recorded RGB.
                    colors=repmat([.43 .60 .67],size(xyz,1),1);
                    description='RGB unavailable - solid fallback';
                end
            case 3
                % Use measured axial camera Z (not Euclidean camera range).
                colors=xyz(:,3);
                zs=sort(colors);n=numel(zs);
                lo=zs(max(1,round(.02*n)));
                hi=zs(min(n,max(1,round(.98*n))));
                if hi<=lo,hi=lo+1;end
                depthLims=[lo hi];
            otherwise
                colors=repmat([.43 .60 .67],size(xyz,1),1);
        end
    end

    function str=onoff(v)
        if v,str='on';else,str='off';end
    end

    function changeSelection(~,~)
        selectionA=get(menuA,'Value');selectionB=get(menuB,'Value');
        refresh();
    end

    function changeAverage(~,~)
        % Arbitrary user-selected N, not restricted to 3/5/7.
        % Require an integer >=1. N can exceed currently available frames;
        % the causal mean then uses only the history that exists.
        rawValue=strtrim(get(avgEdit,'String'));
        proposed=str2double(rawValue);
        if ~isscalar(proposed) || ~isfinite(proposed) || proposed<1 || ...
                proposed~=floor(proposed) || proposed>1e6
            set(avgEdit,'String',num2str(movingWindow));
            warndlg('Enter a whole number N from 1 to 1,000,000. The previous window remains active.', ...
                'Invalid moving-average window');
            return;
        end
        movingWindow=double(proposed);
        set(avgEdit,'String',sprintf('%.0f',movingWindow));
        refresh();
    end

    function a=trailingAverage(v,window)
        % Causal mean over at most N consecutive recorded frames (current
        % frame and the preceding N-1); missing values inside the window
        % are excluded. Never fills an
        % originally invalid measurement or uses future samples.
        a=nan(size(v));
        for i=1:numel(v)
            if ~isfinite(v(i)),continue;end
            block=v(max(1,i-window+1):i);
            block=block(isfinite(block));
            if ~isempty(block),a(i)=mean(block);end
        end
    end

    function av=trailingASI(raw,regionName)
        % Smoothing compares like-with-like: valid bilateral pair IDs must
        % match the current frame. A missing pair does not silently alter
        % the formula underlying the moving average.
        av=nan(size(raw));
        for kk=1:numel(raw)
            if ~isfinite(raw(kk)),continue;end
            cur=measurements(kk);
            desired=false(1,6);
            switch regionName
                case 'UPPER', chosen=upperIdx;
                case 'LOWER', chosen=lowerIdx;
                otherwise,chosen=1:6;
            end
            for pp=chosen
                desired(pp)=selectedPairs(pp)&&cur.pairs(pp).valid;
            end
            vals=[];
            for jj=max(1,kk-movingWindow+1):kk
                if ~isfinite(raw(jj)),continue;end
                other=false(1,6);
                for pp=chosen
                    other(pp)=selectedPairs(pp)&&measurements(jj).pairs(pp).valid;
                end
                if isequal(other,desired)
                    vals(end+1)=raw(jj); %#ok<AGROW>
                end
            end
            if ~isempty(vals),av(kk)=mean(vals);end
        end
    end

    function d=selectedDistanceSeries()
        d=nan(nFrames,1);
        if selectionA==selectionB,return;end
        for k=1:nFrames
            a=squeeze(processed(k,selectionA,:));
            b=squeeze(processed(k,selectionB,:));
            if all(isfinite([a(:);b(:)]))
                d(k)=norm(a-b); % full 3D Euclidean norm, NOT to midline
            end
        end
    end

    function renderSelectedPair(p)
        a=p(selectionA,:);b=p(selectionB,:);
        goodA=all(isfinite(a));goodB=all(isfinite(b));
        showMarks=strcmp(viewMode,'measure') && get(cbLM,'Value')~=0;
        set(selectedMarkA,'Visible',onoff(goodA && showMarks));
        set(selectedMarkB,'Visible',onoff(goodB && showMarks));
        if goodA
            set(selectedMarkA,'XData',a(1),'YData',a(2),'ZData',a(3), ...
                'UserData',struct('kind','Facial landmark','pointIDs',selectionA));
        end
        if goodB
            set(selectedMarkB,'XData',b(1),'YData',b(2),'ZData',b(3), ...
                'UserData',struct('kind','Facial landmark','pointIDs',selectionB));
        end
        show=get(cbLines,'Value') && strcmp(viewMode,'measure') && ...
            goodA && goodB && selectionA~=selectionB;
        set([selectedLineA selectedLineB selectedDistanceLabel], ...
            'Visible',onoff(show));
        if ~show,return;end
        mid=(a+b)/2;
        setLine(selectedLineA,a,mid);
        setLine(selectedLineB,mid,b);
        set(selectedDistanceLabel,'Position',mid+[1 1 0], ...
            'String',sprintf('A-B = %.3f mm',norm(a-b)));
    end

    function renderMeasurementGraph()
        if nFrames==0,return;end
        s=selectedDistanceSeries();avg=trailingAverage(s,movingWindow);
        set(measureTrace,'XData',frameNumbers,'YData',s, ...
            'UserData',struct('kind','Selected distance', ...
            'name',sprintf('%s -> %s',IDs{selectionA},IDs{selectionB}), ...
            'units','mm','landmark',0));
        set(measureDots,'XData',frameNumbers,'YData',s, ...
            'UserData',struct('kind','Selected distance', ...
            'name',sprintf('%s -> %s',IDs{selectionA},IDs{selectionB}), ...
            'units','mm','landmark',0));
        set(measureAvg,'XData',frameNumbers,'YData',avg, ...
            'Visible',onoff(get(cbAvg,'Value')~=0));
        if get(cbAvg,'Value'),displayed=[s;avg];else,displayed=s;end
        ylim(measureAx,smartLimitSigned(displayed));xlim(measureAx,smartX());
        set(measureCursor,'XData',[frameNumbers(frame) frameNumbers(frame)], ...
            'YData',ylim(measureAx));
        u=s(1:frame);u=u(isfinite(u));allvalid=s(isfinite(s));
        current=s(frame);
        if selectionA==selectionB
            titletext='Select two DISTINCT landmarks';
        else
            titletext=sprintf('%s -> %s', ...
                landmarkShortNames{selectionA},landmarkShortNames{selectionB});
        end
        title(measureAx,titletext,'Color',fg,'Interpreter','none','FontSize',10);
        if isempty(allvalid)
            stats='No frames with both landmarks; missing values are NaN.';
        else
            stats=sprintf('Full burst: n=%d/%d | mean %.3f mm | SD %s mm | range %.3f mm', ...
                numel(allvalid),nFrames,mean(allvalid),formatted(safeSD(allvalid)), ...
                max(allvalid)-min(allvalid));
        end
        regionalRaw=[measurements(frame).upper.ASI, ...
            measurements(frame).lower.ASI, measurements(frame).overall.ASI];
        aU=arrayfun(@(m)m.upper.ASI,measurements(:));
        aL=arrayfun(@(m)m.lower.ASI,measurements(:));
        aO=arrayfun(@(m)m.overall.ASI,measurements(:));
        sU=trailingASI(aU(:),'UPPER');
        sL=trailingASI(aL(:),'LOWER');
        sO=trailingASI(aO(:),'OVERALL');
        if get(cbAvg,'Value')
            summary=sprintf('ASI mean N=%d: U %s | L %s | O %s (raw %s/%s/%s)', ...
                movingWindow,formatted(sU(frame)),formatted(sL(frame)), ...
                formatted(sO(frame)),formatted(regionalRaw(1)), ...
                formatted(regionalRaw(2)),formatted(regionalRaw(3)));
        else
            summary=sprintf('ASI RAW: Upper %s | Lower %s | Overall %s', ...
                formatted(regionalRaw(1)),formatted(regionalRaw(2)), ...
                formatted(regionalRaw(3)));
        end
        set(measureNote,'String',{summary; ...
            sprintf('A [%s]  vs  B [%s]',IDs{selectionA},IDs{selectionB}); ...
            sprintf('Actual 3D line: sqrt((X_A-X_B)^2+(Y_A-Y_B)^2+(Z_A-Z_B)^2)'); ...
            sprintf('Frame %d: %s mm | Moving mean N=%d: %s mm', ...
            frameNumbers(frame),formatted(current),movingWindow,formatted(avg(frame)))});
        if isempty(u),rm=NaN;else,rm=mean(u);end
        set(measureStats,'String',{stats; ...
            sprintf('Through frame %d: mean %s mm | sample SD %s mm (n=%d)', ...
            frameNumbers(frame),formatted(rm),formatted(safeSD(u)),numel(u))});
    end

    function toggleViewMode(~,~)
        if strcmp(viewMode,'measure'),viewMode='summary';
        else,viewMode='measure';end
        refresh();
    end

    function openMeasurementGraph(~,~)
        if nFrames==0,return;end
        s=selectedDistanceSeries();avg=trailingAverage(s,movingWindow);
        f=figure('Name','Large A-B measurement plot and 3D XYZ audit', ...
            'NumberTitle','off','Color','w','Position',[80 70 1300 760]);
        a=axes('Parent',f,'Units','normalized','Position',[.065 .41 .9 .49]);
        hold(a,'on');grid(a,'on');
        plot(a,frameNumbers,s,'-o','Color',selectionColorA,'MarkerFaceColor',selectionColorB, ...
            'LineWidth',1.8,'MarkerSize',5,'DisplayName','Raw A-B Euclidean distance');
        plot(a,frameNumbers,avg,'--','Color',[.1 .6 .4], ...
            'LineWidth',1.5,'DisplayName','Causal moving mean');
        xlim(a,smartX());ylim(a,smartLimitSigned([s;avg]));
        xlabel(a,'Frame');ylabel(a,'3D distance (mm)');legend(a,'Location','best');
        title(a,sprintf('%s -> %s',IDs{selectionA},IDs{selectionB}), ...
            'Interpreter','none');
        info=cell(nFrames,11);
        for k=1:nFrames
            ap=squeeze(processed(k,selectionA,:));bp=squeeze(processed(k,selectionB,:));
            info(k,:)={frameNumbers(k),s(k),avg(k),all(isfinite(ap)), ...
                all(isfinite(bp)),ap(1),ap(2),ap(3),bp(1),bp(2),bp(3)};
        end
        uitable(f,'Units','normalized','Position',[.065 .06 .9 .29], ...
            'Data',info,'ColumnName',{'Frame','Distance_mm','CausalMean_mm', ...
            'A_valid','B_valid','A_X_mm','A_Y_mm','A_Z_mm', ...
            'B_X_mm','B_Y_mm','B_Z_mm'});
    end

    function openEnlarged3D(~,~)
        if nFrames==0,return;end
        c=rawClouds{frame};p=squeeze(processed(frame,:,:));
        f=figure('Name',sprintf('Large rotating 3D cloud | frame %d',frameNumbers(frame)), ...
            'NumberTitle','off','Color',[.045 .07 .1], ...
            'Position',[95 45 1280 870]);
        a=axes('Parent',f,'Position',[.07 .10 .85 .83], ...
            'Color',[.03 .05 .08],'XColor','w','YColor','w','ZColor','w');
        hold(a,'on');grid(a,'on');axis(a,'equal');axis(a,'vis3d');
        style=get(cloudModeMenu,'Value');
        [pts,pointColors,depthLimits,cloudDescription]=cloudColors(c,style);
        if get(cbCloud,'Value') && ~isempty(pts)
            scatter3(a,pts(:,1),pts(:,2),pts(:,3),4,pointColors,'filled', ...
                'MarkerEdgeColor','none');
            if ~isempty(depthLimits)
                colormap(a,parula(256));caxis(a,depthLimits);
            end
        end
        drawAll=(style~=1)||(get(cbAllLM,'Value')~=0);
        if drawAll && get(cbLM,'Value')
            if style==1
                valid=all(isfinite(p),2);
                scatter3(a,p(valid,1),p(valid,2),p(valid,3), ...
                    65,[.70 .77 .81],'filled','MarkerEdgeColor','k');
            else
                groups={upperDisplayIDs,midDisplayIDs,lowerDisplayIDs};
                tints={upperColor,midColor,lowerColor};
                for gg=1:3
                    ids=groups{gg};valid=all(isfinite(p(ids,:)),2);
                    points=p(ids(valid),:);
                    if ~isempty(points)
                        scatter3(a,points(:,1),points(:,2),points(:,3), ...
                            72,tints{gg},'filled','MarkerEdgeColor','w', ...
                            'LineWidth',.9);
                    end
                end
                other=setdiff(1:nLM,[upperDisplayIDs midDisplayIDs lowerDisplayIDs]);
                valid=all(isfinite(p(other,:)),2);points=p(other(valid),:);
                if ~isempty(points)
                    scatter3(a,points(:,1),points(:,2),points(:,3), ...
                        60,[1 .89 .39],'filled','MarkerEdgeColor','k');
                end
            end
        end
        if get(cbDenseLM,'Value') && get(cbLM,'Value') && any(isfinite(denseXYZ(:)))
            groups={denseEyeIDs,denseBrowIDs,denseNoseIDs};
            tints={[1 .77 .22],[1 .38 .44],[.28 .96 .70]};
            for gg=1:numel(groups)
                [tf,ii]=ismember(groups{gg},denseIDs);ii=ii(tf);
                denseP=reshape(denseXYZ(frame,ii,:),[],3);
                denseP=denseP(all(isfinite(denseP),2),:);
                if ~isempty(denseP)
                    scatter3(a,denseP(:,1),denseP(:,2),denseP(:,3), ...
                        24,tints{gg},'filled');
                end
            end
        end
        if get(cbText,'Value') && get(cbLM,'Value')
            for j=1:nLM
                if all(isfinite(p(j,:))) && (drawAll||j==selectionA||j==selectionB)
                    text(a,p(j,1)+1,p(j,2)+1,p(j,3),IDs{j}, ...
                        'Interpreter','none','Color','w','FontSize',8);
                end
            end
        end
        pa=p(selectionA,:);pb=p(selectionB,:);
        if get(cbLM,'Value') && strcmp(viewMode,'measure')
            if all(isfinite(pa)),scatter3(a,pa(1),pa(2),pa(3),240,selectionColorA,'filled');end
            if all(isfinite(pb)),scatter3(a,pb(1),pb(2),pb(3),240,selectionColorB,'filled');end
        end
        if get(cbLines,'Value') && all(isfinite([pa pb])) && selectionA~=selectionB
            halfway=(pa+pb)/2;
            plot3(a,[pa(1) halfway(1)],[pa(2) halfway(2)], ...
                [pa(3) halfway(3)],'-','Color',selectionColorA,'LineWidth',3);
            plot3(a,[halfway(1) pb(1)],[halfway(2) pb(2)], ...
                [halfway(3) pb(3)],'-','Color',selectionColorB,'LineWidth',3);
            text(a,halfway(1),halfway(2),halfway(3), ...
                sprintf('  %.3f mm',norm(pa-pb)),'Color','w','FontSize',12);
        end
        xlabel(a,'Camera X (mm)');ylabel(a,'Camera Y (mm)');zlabel(a,'Camera Z (mm)');
        set(a,'YDir','reverse');view(a,0,90);
        nValid=sum(all(isfinite(p),2));
        title(a,sprintf('%s | Frame %d | %d/%d recorded 3D dots (drag to rotate)', ...
            cloudDescription,frameNumbers(frame),nValid,nLM), ...
            'Color','w','Interpreter','none');rotate3d(f,'on');
    end

    function setLine(h,a,b)
        set(h,'XData',[a(1),b(1)],'YData',[a(2),b(2)], ...
            'ZData',[a(3),b(3)],'Visible','on');
    end

    function assignMarks(h,indexes,show)
        p=squeeze(processed(frame,:,:));
        if ~show||isempty(indexes),set(h,'Visible','off');return;end
        ok=all(isfinite(p(indexes,:)),2);pts=p(indexes(ok),:);
        set(h,'XData',pts(:,1),'YData',pts(:,2),'ZData',pts(:,3), ...
            'UserData',struct('kind','Facial landmark', ...
            'pointIDs',indexes(ok)), ...
            'Visible','on');
    end

    function [series,legendText,units,whichLM,kind]=selectedGraphSeries(panel,regionSeries)
        % A named single landmark is NOT an ASI: show one physical metric.
        % Pair and regional ASI selections are separate diagnostic entries.
        selection=get(panel.graphMenu,'Value');
        whichLM=0; kind='Landmark'; units='mm';
        if selection<=numel(panel.graphLM)
            whichLM=panel.graphLM(selection);
            metric=get(panel.metricMenu,'Value');
            set(panel.metricMenu,'Enable','on');
            series=nan(nFrames,1);
            for f=1:nFrames
                q=squeeze(localXYZ(f,whichLM,:));
                p=squeeze(processed(f,whichLM,:));
                if metric>=5 && metric<=7
                    if all(isfinite(p)),series(f)=p(metric-4);end
                elseif metric==8
                    if f>=2
                        pa=squeeze(processed(f-1,whichLM,:));
                        if all(isfinite([pa(:);p(:)]))
                            series(f)=norm(p-pa);
                        end
                    end
                elseif all(isfinite(q))
                    switch metric
                        case 1,series(f)=abs(q(1));
                        case 2,series(f)=q(1);
                        case 3,series(f)=q(2);
                        case 4,series(f)=q(3);
                    end
                end
            end
            legendText=sprintf('%s: %s',landmarkShortNames{whichLM}, ...
                landmarkMetricOptions{metric});
        elseif selection==numel(panel.graphLM)+1
            series=regionSeries;
            legendText='Regional ASI TOTAL (pooled landmarks)';
            units='index';kind='Region ASI';
            set(panel.metricMenu,'Enable','off');
        else
            p=selection-numel(panel.graphLM)-1;
            p=min(numel(panel.graphPairIndices),max(1,p));
            pairNo=panel.graphPairIndices(p);
            series=nan(nFrames,1);
            for f=1:nFrames
                t=measurements(f).pairs(pairNo);
                if t.valid && t.width>0
                    series(f)=100*sqrt(t.dH^2+t.dV^2+t.dD^2)/t.width;
                end
            end
            legendText=sprintf('%s left/right pair ASI',pairNames{pairNo});
            units='index';kind='Pair ASI';
            set(panel.metricMenu,'Enable','off');
        end
    end

    function renderIndex(panel,regionSeries,r,name,idx)
        [series,graphLabel,units,whichLM,kind]=selectedGraphSeries(panel,regionSeries);
        averaged=trailingAverage(series,movingWindow);
        if strcmp(kind,'Region ASI')
            averaged=trailingASI(regionSeries,name);
        end
        effectiveRegional=trailingASI(regionSeries,name);
        if get(cbAvg,'Value') && isfinite(effectiveRegional(frame))
            asiHeader=sprintf('%s ASI N=%d mean %.3f | raw %.3f', ...
                name,movingWindow,effectiveRegional(frame),r.ASI);
        else
            asiHeader=sprintf('%s ASI raw %.3f',name,r.ASI);
        end
        set(panel.averageTrace,'XData',frameNumbers,'YData',averaged, ...
            'Visible',onoff(get(cbAvg,'Value')~=0));
        set(panel.trace,'XData',frameNumbers,'YData',series, ...
            'UserData',struct('name',graphLabel,'units',units, ...
            'landmark',whichLM,'kind',kind));
        if whichLM>0,axisTitle=landmarkShortNames{whichLM};
        else,axisTitle=kind;end
        title(panel.a,axisTitle,'FontSize',7,'Interpreter','none','Color',fg);
        ylabel(panel.a,units);
        if get(cbAvg,'Value'),ylimits=smartLimitSigned([series;averaged]);
        else,ylimits=smartLimitSigned(series);end
        ylim(panel.a,ylimits);xlim(panel.a,smartX());
        set(panel.cursor,'XData',[frameNumbers(frame) frameNumbers(frame)], ...
            'YData',ylimits);
        v=series(1:frame);v=v(isfinite(v));
        if isempty(v),mu='N/A';else,mu=sprintf('%.3f',mean(v));end
        if numel(v)<2,sd='N/A';else,sd=sprintf('%.3f',std(v,0));end
        set(panel.current,'String',{sprintf('Frame %d: %s %s', ...
            frameNumbers(frame),formatted(series(frame)),units), ...
            sprintf('Graph mean %s | SD %s',mu,sd)});
        labels=getappdata(panel.p,'DotLabels');
        if ~isempty(labels)
            validHandles=isgraphics(labels);
            delete(labels(validHandles));
        end
        labels=gobjects(0,1);
        if isgraphics(cbDotLabels)&&get(cbDotLabels,'Value')
            valid=find(isfinite(series));
            labels=gobjects(numel(valid),1);
            yrange=ylimits(2)-ylimits(1);
            for h=1:numel(valid)
                ii=valid(h);
                yoff=(-1)^ii*0.045*yrange;
                labels(h)=text(panel.a,frameNumbers(ii), ...
                    series(ii)+yoff,sprintf('%.1f',series(ii)), ...
                    'Color',fg,'FontSize',6, ...
                    'HorizontalAlignment','center','Clipping','on', ...
                    'HitTest','off','Interpreter','none');
            end
        end
        setappdata(panel.p,'DotLabels',labels);
        % The left text retains pooled ASI, with raw and trailing-N mean
        % explicitly distinguished; an individual trace is never an ASI.
        if r.valid
            detail=sprintf([ ...
                '%s   pairs %d/%d\n' ...
                'H raw 100*(%.2f/%.2f) = %.3f\n' ...
                'V raw 100*(%.2f/%.2f) = %.3f\n' ...
                'D raw 100*(%.2f/%.2f) = %.3f\n' ...
                'M 100*(%.2f/%.2f) = %.3f\n' ...
                'TOTAL=sqrt(H^2+V^2+D^2+M^2)\n' ...
                'Raw bilateral ASI = %.3f\n' ...
                'Individual graph: %s\n' ...
                '%s'], ...
                asiHeader,r.count,r.possible, ...
                r.sumH,r.sumW,r.H,r.sumV,r.sumW,r.V, ...
                r.sumD,r.sumW,r.D,r.sumM,r.sumW,r.M, ...
                r.bilateralASI,kind,shortPairs(idx));
        else
            detail=sprintf([ ...
                '%s TOTAL ASI: unavailable\n' ...
                'Pairs %d/%d (need >=1 each region)\n' ...
                'H,V,D=100*SUM(delta)/SUM(width)\n' ...
                'M=100*SUM(2*|midline U|)/SUM(width)\n' ...
                'TOTAL=sqrt(H^2+V^2+D^2+M^2)\n' ...
                'Missing/unselected pairs are NOT zero\n' ...
                'Graph: %s\n' ...
                '%s'],name,r.count,r.possible,kind,shortPairs(idx));
        end
        set(panel.t,'String',detail);
    end

    function ylimits=smartLimitSigned(v)
        t=v(isfinite(v));
        if isempty(t),ylimits=[0 1];return;end
        mn=min(t);mx=max(t);padding=max(.25,.14*max(mx-mn,1));
        ylimits=[mn-padding mx+padding];
        if mn>=0,ylimits(1)=max(0,ylimits(1));end
        if diff(ylimits)<eps,ylimits(2)=ylimits(1)+1;end
    end

    function toggleDotInspection(~,~)
        % Use a conventional MATLAB data cursor; avoids trying to infer
        % values from pixel positions. Rotation can be re-enabled above.
        try
            rotate3d(fig,'off');pan(fig,'off');zoom(fig,'off');
            dcm=datacursormode(fig);
            if strcmp(get(dcm,'Enable'),'on')
                set(dcm,'Enable','off');
            else
                set(dcm,'DisplayStyle','datatip', ...
                    'UpdateFcn',@individualDotTip,'Enable','on');
            end
        catch ME
            warndlg(sprintf('Unable to enable dot inspection: %s',ME.message), ...
                'Graph data tips');
        end
    end

    function out=individualDotTip(~,evt)
        % Click a graph marker: shows its exact stored Y and frame value.
        h=get(evt,'Target'); pos=get(evt,'Position');
        ix=get(evt,'DataIndex');
        z=[];
        if isprop(h,'UserData'),z=get(h,'UserData');end
        % An individual point on the 3D face or cloud has XYZ coordinates,
        % rather than a frame-by-frame graph-value interpretation.
        if isstruct(z)&&isfield(z,'kind') ...
                && (strcmp(z.kind,'Facial landmark')||strcmp(z.kind,'Cloud point') ...
                ||strcmp(z.kind,'Dense mesh vertex'))
            out={'3D point under cursor', ...
                sprintf('Camera X: %.4f mm',pos(1)), ...
                sprintf('Camera Y: %.4f mm',pos(2))};
            if numel(pos)>=3
                out{end+1}=sprintf('Camera Z: %.4f mm',pos(3));
            end
            if strcmp(z.kind,'Facial landmark') && ...
                    isfield(z,'pointIDs') && ix>=1 && ix<=numel(z.pointIDs)
                j=z.pointIDs(ix);
                out{end+1}=sprintf('%s [%s]',landmarkShortNames{j},IDs{j});
                q=squeeze(localXYZ(frame,j,:));
                if all(isfinite(q))
                    out{end+1}=sprintf('U %.3f | V %.3f | W %.3f mm',q);
                    out{end+1}=sprintf('|U| to midline: %.3f mm',abs(q(1)));
                end
                out{end+1}='Cloud-snapped label; anatomical identity unverified';
            elseif strcmp(z.kind,'Dense mesh vertex')
                if isfield(z,'meshIDs') && ix>=1 && ix<=numel(z.meshIDs)
                    out{end+1}=sprintf('MediaPipe vertex ID %d',z.meshIDs(ix));
                end
                out{end+1}='Mesh XYZ is from same-frame depth; anatomy not independently confirmed';
            elseif strcmp(z.kind,'Cloud point')
                out{end+1}=sprintf('Point-cloud row: %d',ix);
                out{end+1}='Not necessarily an anatomical landmark';
            end
            return;
        end
        if isempty(ix)||~isfinite(ix)||ix<1||ix>nFrames
            out={sprintf('X: %.3f',pos(1)),sprintf('Y: %.3f',pos(2))};return;
        end
        out={sprintf('Frame %d',frameNumbers(ix)), ...
            sprintf('Plot value %.4f',pos(2))};
        if isstruct(z)&&isfield(z,'name')
            out{end+1}=z.name;
            out{end+1}=sprintf('Value: %.5f %s',pos(2),z.units);
            if isfield(z,'landmark')&&z.landmark>0
                j=z.landmark;
                q=squeeze(localXYZ(ix,j,:));
                p=squeeze(processed(ix,j,:));
                out{end+1}=sprintf('ID: %s',IDs{j});
                if all(isfinite(q))
                    out{end+1}=sprintf('U %.3f | V %.3f | W %.3f mm',q);
                else
                    out{end+1}='Face-local coordinates: unavailable';
                end
                if all(isfinite(p))
                    out{end+1}=sprintf('Camera X %.3f, Y %.3f, Z %.3f mm',p);
                end
            end
        end
    end

    function openIndividualGraphTable(grp)
        if nFrames==0,return;end
        switch grp
            case 1,panel=panelU;region='Upper';
                rs=arrayfun(@(m)m.upper.ASI,measurements);
            case 2,panel=panelL;region='Lower';
                rs=arrayfun(@(m)m.lower.ASI,measurements);
            otherwise,panel=panelO;region='Overall';
                rs=arrayfun(@(m)m.overall.ASI,measurements);
        end
        [series,graphLabel,units,j,~]=selectedGraphSeries(panel,rs(:));
        f=figure('Name',[region ' | individual dot values | ' graphLabel], ...
            'NumberTitle','off','Color','w','Position',[90 80 1220 700]);
        axFull=axes('Parent',f,'Units','normalized', ...
            'Position',[.065 .39 .91 .51]);hold(axFull,'on');grid(axFull,'on');
        plot(axFull,frameNumbers,series,'-o','LineWidth',1.4,'MarkerSize',5);
        xlim(axFull,smartX());ylim(axFull,smartLimitSigned(series));
        xlabel(axFull,'Frame');ylabel(axFull,units);
        title(axFull,graphLabel,'Interpreter','none');
        data=cell(nFrames,9);
        for k=1:nFrames
            data{k,1}=frameNumbers(k);
            if isfinite(series(k)),data{k,2}=series(k);else,data{k,2}=NaN;end
            data{k,3}=isfinite(series(k));
            if j>0
                p=squeeze(processed(k,j,:));q=squeeze(localXYZ(k,j,:));
                data(k,4:6)=num2cell(p(:)');
                data(k,7:9)=num2cell(q(:)');
            else
                data(k,4:9)=num2cell(nan(1,6));
            end
        end
        uitable(f,'Units','normalized','Position',[.065 .065 .91 .265], ...
            'Data',data,'ColumnName',{'Frame',['Value (' units ')'], ...
            'Valid','Camera X mm','Camera Y mm','Camera Z mm', ...
            'Face U mm','Face V mm','Face W mm'});
        uicontrol(f,'Style','text','Units','normalized', ...
            'Position',[.065 .93 .9 .05],'String', ...
            'Each row is one displayed dot. NaN = unavailable; a single landmark value is NOT an ASI.', ...
            'HorizontalAlignment','left','BackgroundColor','w');
    end

    function s=shortPairs(idx)
        if isempty(idx),s='(none selected)';else,s=strjoin(pairNames(idx),', ');end
    end

    function a=smartX()
        if nFrames<=1,a=[1 2];else,a=[frameNumbers(1) frameNumbers(end)];end
    end

    function a=smartLimit(v)
        ok=v(isfinite(v));
        if isempty(ok),a=[0 1];return;end
        ymin=max(0,min(ok));ymax=max(ok);
        padding=max(.25,(ymax-ymin)*.18);
        a=[max(0,ymin-padding) ymax+padding];
        if a(1)==a(2),a(2)=a(1)+1;end
    end

    function s=formatted(x)
        if isfinite(x),s=sprintf('%.3f',x);else,s='N/A';end
    end

    function changeSmoothing(~,~)
        % This only reduces visible jitter. It is NOT anatomical correction.
        stopPlayback();smoothMode=get(cbSmooth,'Value')~=0;
        recalcWithMode();refresh();
    end

    function chooseLandmarks(~,~)
        stopPlayback();
        sel=figure('Name','Landmark selection - experimental', ...
            'NumberTitle','off','Color',[.12 .16 .22], ...
            'Position',[120 100 920 700],'MenuBar','none','ToolBar','none', ...
            'WindowStyle','modal');
        uicontrol(sel,'Style','text','Units','normalized', ...
            'Position',[.03 .89 .94 .09],'BackgroundColor',[.12 .16 .22], ...
            'ForegroundColor','w','FontSize',10,'HorizontalAlignment','left', ...
            'String',{ 'Select bilateral pairs for H/V/D scores; optional single midline points contribute M only.', ...
            'NASION and SUBNASALE anchor the midline (M=0 by definition); select them for display, not independent score.', ...
            'Unavailable anatomy is listed below but cannot be synthesized from this recording.'});
        up=uipanel(sel,'Units','normalized','Position',[.03 .21 .45 .66], ...
            'Title','UPPER: forehead, brow and ocular region', ...
            'BackgroundColor',[.14 .18 .25],'ForegroundColor',upperColor);
        lo=uipanel(sel,'Units','normalized','Position',[.51 .21 .46 .66], ...
            'Title','LOWER: nasal, zygomatic, lips and chin', ...
            'BackgroundColor',[.14 .18 .25],'ForegroundColor',lowerColor);
        controlsP=gobjects(6,1);controlsM=gobjects(nLM,1);
        for z=1:3
            controlsP(z)=option(up,sprintf('[PAIR] %s L/R',pairNames{z}), ...
                .96-(z-1)*.085,selectedPairs(z));
        end
        for z=1:numel(upperMid)
            idx=upperMid(z);
            controlsM(idx)=option(up,sprintf('[MID] %s',IDs{idx}), ...
                .68-(z-1)*.085,selectedMid(idx));
        end
        addUnavailable(up,{'Trichion (true hairline): not recorded', ...
            'Upper/lower eyelid margins: not recorded'},.38);
        for z=4:6
            tag='';if z==6,tag=' [L missing]';end
            controlsP(z)=option(lo,sprintf('[PAIR] %s L/R%s',pairNames{z},tag), ...
                .96-(z-4)*.071,selectedPairs(z));
        end
        for z=1:numel(lowerMid)
            idx=lowerMid(z);
            controlsM(idx)=option(lo,sprintf('[MID] %s',IDs{idx}), ...
                .73-(z-1)*.067,selectedMid(idx));
        end
        addUnavailable(lo,{'Crista philtri / Cupid bow L/R: unavailable', ...
            'Vermilion border curve: unavailable', ...
            'Stomion: unavailable'},.29);
        uicontrol(sel,'Style','text','Units','normalized','Position',[.03 .13 .94 .06], ...
            'String','WARNING: selecting midline points changes TOTAL definition. Compare scans only with identical settings.', ...
            'BackgroundColor',[.12 .16 .22],'ForegroundColor',[1 .80 .5]);
        uicontrol(sel,'Style','pushbutton','Units','normalized', ...
            'Position',[.37 .035 .26 .075],'String','Apply selections', ...
            'FontWeight','bold','Callback',@applySelection);
        function h=option(parent,label,y,val)
            h=uicontrol(parent,'Style','checkbox','Units','normalized', ...
                'Position',[.04 y-.035 .94 .065],'String',label,'Value',double(val), ...
                'BackgroundColor',[.14 .18 .25],'ForegroundColor','w','FontSize',9);
        end
        function addUnavailable(parent,lines,y)
            uicontrol(parent,'Style','text','Units','normalized', ...
                'Position',[.04 y-.16 .92 .17], ...
                'String',lines,'HorizontalAlignment','left', ...
                'BackgroundColor',[.14 .18 .25], ...
                'ForegroundColor',[1 .7 .5],'FontSize',9);
        end
        function applySelection(~,~)
            for zz=1:6
                selectedPairs(zz)=get(controlsP(zz),'Value')~=0;
            end
            for zz=[upperMid(:)' lowerMid(:)']
                selectedMid(zz)=get(controlsM(zz),'Value')~=0;
            end
            if ~any(selectedPairs(upperIdx)) || ~any(selectedPairs(lowerIdx))
                errordlg('Select at least one UPPER and one LOWER bilateral pair.', ...
                    'Missing bilateral measurements');return;
            end
            recomputeAll();
            if isgraphics(sel),delete(sel);end
            refresh();
        end
    end

    function landmarkAudit(~,~)
        if nFrames==0,return;end
        % Explicit warnings: visual/speed QC cannot confirm anatomy.
        yy=nan(nFrames,nLM,3);
        for f=1:nFrames,yy(f,:,:)=rawSeeds(f,:,:);end
        y=squeeze(yy(frame,:,:));
        status=cell(nLM,1);displace=nan(nLM,1);
        for i=1:nLM
            if ~all(isfinite(y(i,:))),status{i}='MISSING';continue;end
            status{i}='Recorded; NOT manually verified';
            if frame>1
                a=squeeze(yy(frame-1,i,:))';
                if all(isfinite(a)),displace(i)=norm(y(i,:)-a);end
            end
            if isfinite(displace(i))&&displace(i)>15
                status{i}='REVIEW: >15 mm raw jump';
            end
        end
        if all(isfinite(y([7 8 9 14],:)),'all')
            % Check known issues without automatically relocating anatomy.
            if abs(y(14,3)-y(7,3))>60
                status{14}='REVIEW: MENTON depth far from GNATHION';
            end
        end
        if all(isfinite(y([3 4 7 8 9],:)),'all')
            cm=mean(y([3 4],2));
            if y(8,2)<=cm,status{8}='REVIEW LI: above mouth-corner midpoint';end
            if y(9,2)>=y(8,2),status{9}='REVIEW LS: not above LI';end
        end
        ids=IDs(:);
        audit=table(ids,y(:,1),y(:,2),y(:,3),displace,status, ...
            'VariableNames',{'Landmark','X_mm','Y_mm','Z_mm','RawStep_mm','QualityNote'});
        af=figure('Name','Landmark audit - no anatomical ground truth', ...
            'NumberTitle','off','Color','w','Position',[140 70 1090 740]);
        uitable(af,'Units','normalized','Position',[.02 .11 .96 .86], ...
            'Data',table2cell(audit),'ColumnName',audit.Properties.VariableNames);
        uicontrol(af,'Style','text','Units','normalized', ...
            'Position',[.025 .015 .95 .085],'BackgroundColor','w', ...
            'HorizontalAlignment','left', ...
            'String',['X/Y/Z are ORIGINAL seeded coordinates. RawStep is an ' ...
            'unregistered camera-space step, not actual lip speed. ' ...
            'All positions require manual annotation / annotated validation.']);
    end

    function changeFPS(~,~)
        val=strtrim(get(fpsEdit,'String'));
        if isempty(val),fps=NaN;refresh();return;end
        candidate=str2double(val);
        if ~isfinite(candidate)||candidate<=0
            errordlg('FPS must be blank (unknown) or a positive, measured value.', ...
                'Invalid source FPS');
            if isfinite(fps),set(fpsEdit,'String',num2str(fps));
            else,set(fpsEdit,'String','');end
            return;
        end
        fps=candidate;refresh();
    end

    function slideFrame(~,~)
        stopPlayback();
        frame=min(nFrames,max(1,round(get(slider,'Value'))));refresh();
    end

    function togglePlay(~,~)
        if playing,stopPlayback();return;end
        playing=true;set(startButton,'String','Pause');
        try
            if strcmp(clockTimer.Running,'off'),start(clockTimer);end
        catch ME
            stopPlayback();errordlg(ME.message,'Unable to start timer');
        end
    end

    function stopPlayback()
        playing=false;
        if hasProvider,set(startButton,'String','Start LIVE feed');
        else,set(startButton,'String',sprintf('Play %d frames',nFrames));end
        if exist('clockTimer','var') && isvalid(clockTimer) ...
        && strcmp(clockTimer.Running,'on'),stop(clockTimer);end
    end

    function timerTick(~,~)
        if ~playing||~isgraphics(fig),return;end
        if ~hasProvider
            frame=mod(frame,nFrames)+1;refresh();
            return;
        end
        try
            s=opt.FrameProvider();
            if isempty(s),return;end
            if ~isstruct(s)||~isfield(s,'cloud')||size(s.cloud,2)<3
                error('FrameProvider must return a struct with Nx3 or Nx6 cloud.');
            end
            if ~isnumeric(s.cloud)||size(s.cloud,2)~=3&&size(s.cloud,2)~=6
                error('cloud must be a numeric Nx3 or Nx6 matrix.');
            end
            if isfield(s,'landmarks')&&~isempty(s.landmarks)
                seed=s.landmarks;source='Live detector landmarks';
                if ~isnumeric(seed)||~isequal(size(seed),[nLM 3])
                    error('landmarks must be 21x3 in D.landmarkIDs order.');
                end
            elseif nFrames>0
                seed=squeeze(processed(end,:,:));
                source='Propagated NN (UNVALIDATED)';
            else
                error(['Initial live frame must include 21x3 labeled XYZ. ' ...
                    'Point-cloud-only identification is not implemented.']);
            end
            timestamp=NaN;
            if isfield(s,'timestampSeconds') && isnumeric(s.timestampSeconds) ...
                    && isscalar(s.timestampSeconds) && isfinite(s.timestampSeconds)
                timestamp=double(s.timestampSeconds);
            end
            [p,q,m]=processFrame(s.cloud,seed);
            dFrame=nan(1,numel(denseIDs),3);
            if isfield(s,'denseXYZ') && isnumeric(s.denseXYZ) && ...
                    isequal(size(s.denseXYZ),[numel(denseIDs) 3])
                if ~isfield(s,'denseIDs') || ...
                        isequal(double(s.denseIDs(:)'),denseIDs)
                    dFrame=reshape(double(s.denseXYZ),1,numel(denseIDs),3);
                end
            end
            denseXYZ(end+1,:,:)=dFrame;
            rawClouds{end+1,1}=double(s.cloud);
            rawSeeds(end+1,:,:)=reshape(double(seed),1,nLM,3);
            processed(end+1,:,:)=reshape(p,1,nLM,3);
            rawProcessed(end+1,:,:)=reshape(p,1,nLM,3);
            localXYZ(end+1,:,:)=reshape(q,1,nLM,3);
            acquiredTime(end+1,1)=timestamp;
            measurements(end+1,1)=m;
            frameSource{end+1,1}=source;
            totalReceived=totalReceived+1;
            frameNumbers(end+1,1)=totalReceived;
            if numel(rawClouds)>opt.MaxLiveFrames
                denseXYZ(1,:,:)=[];
                rawClouds(1)=[];rawSeeds(1,:,:)=[];
                processed(1,:,:)=[];localXYZ(1,:,:)=[];
                rawProcessed(1,:,:)=[];
                acquiredTime(1)=[];measurements(1)=[];
                frameSource(1)=[];frameNumbers(1)=[];
            end
            nFrames=numel(rawClouds);frame=nFrames;
            smoothProcessed=causalSmooth(rawProcessed);
            recalcWithMode();
            if nFrames==1||mod(totalReceived,20)==0,fitCloud();end
            refresh();
        catch ME
            stopPlayback();
            warning('FacialASI:LiveProvider','Live stream paused: %s',ME.message);
            errordlg(ME.message,'Live stream paused');
        end
    end

    function fitCloud()
        if nFrames==0,return;end
        % Use first / central / last frames to get robust fixed axes;
        % do NOT clip actual point clouds or modify measurement statistics.
        inds=unique([1 round(nFrames/2) nFrames]);P=[];
        for k=inds
            c=rawClouds{k};
            if isempty(c),continue;end
            good=all(isfinite(c(:,1:3)),2);
            pts=c(good,1:3);P=[P;pts(1:8:end,:)]; %#ok<AGROW>
        end
        if isempty(P),return;end
        lim=nan(3,2);
        for j=1:3
            s=sort(P(:,j));nn=numel(s);
            lim(j,:)=[s(max(1,round(.01*nn))) s(min(nn,max(1,round(.99*nn))))];
        end
        span=max(1,lim(:,2)-lim(:,1));pad=max(10,.07*span);
        xlim(ax,lim(1,:)+[-pad(1) pad(1)]);
        ylim(ax,lim(2,:)+[-pad(2) pad(2)]);
        zlim(ax,lim(3,:)+[-pad(3) pad(3)]);
        daspect(ax,[1 1 1]);view(ax,0,90);
    end

    function enableRotate(~,~)
        % Restrict orbit to the 3D face axes, not the right-hand plots.
        pan(fig,'off');zoom(fig,'off');
        try,datacursormode(fig,'off');catch,end
        h=rotate3d(fig);
        try
            setAllowAxesRotate(h,findall(fig,'Type','axes'),false);
            setAllowAxesRotate(h,ax,true);
        catch
            % Older MATLAB versions may not expose axes restrictions.
        end
        set(h,'Enable','on');
    end

    function enablePan(~,~)
        rotate3d(fig,'off');zoom(fig,'off');pan(fig,'on');
    end

    function enableZoom(~,~)
        rotate3d(fig,'off');pan(fig,'off');zoom(fig,'on');
    end

    function sideView(~,~)
        rotate3d(fig,'off');pan(fig,'off');zoom(fig,'off');
        view(ax,90,0);
    end

    function frontView(~,~)
        rotate3d(fig,'off');pan(fig,'off');zoom(fig,'off');
        view(ax,0,90);
    end

    function c=pairTint(j)
        if j<=3,c=upperColor;elseif j==4||j==6,c=midColor;
        else,c=lowerColor;end
    end

    function c=landmarkTint(j)
        if ismember(j,upperDisplayIDs),c=upperColor;
        elseif ismember(j,midDisplayIDs),c=midColor;
        elseif ismember(j,lowerDisplayIDs),c=lowerColor;
        else,c=fg;end
    end

    function openReview(~,~)
        % A non-modal comparison window tied to the active frame slider.
        if nFrames==0,return;end
        if ~isempty(reviewFig)&&isgraphics(reviewFig)
            figure(reviewFig);updateReview();return;
        end
        reviewFig=figure('Name','Bilateral distances to reference sagittal plane', ...
            'NumberTitle','off','Color',[.08 .105 .15], ...
            'Position',[100 60 1260 740]);
        uicontrol(reviewFig,'Style','text','Units','normalized', ...
            'Position',[.025 .93 .25 .045],'String','Review bilateral pair', ...
            'BackgroundColor',[.08 .105 .15],'ForegroundColor',fg, ...
            'HorizontalAlignment','left','FontWeight','bold');
        reviewMenu=uicontrol(reviewFig,'Style','popupmenu','Units','normalized', ...
            'Position',[.265 .925 .26 .055],'String',pairNames,'Value',5, ...
            'BackgroundColor','w','Callback',@updateReview);
        uicontrol(reviewFig,'Style','text','Units','normalized', ...
            'Position',[.54 .93 .275 .045], ...
            'String','Perpendicular distances to U = 0', ...
            'ForegroundColor',fg,'BackgroundColor',[.08 .105 .15], ...
            'HorizontalAlignment','right');
        uicontrol(reviewFig,'Style','pushbutton','Units','normalized', ...
            'Position',[.835 .925 .14 .05],'String','LS / LI QC', ...
            'Callback',@openLipQC);
        reviewAxLR=axes('Parent',reviewFig,'Units','normalized', ...
            'Position',[.075 .46 .87 .385], ...
            'Color',[.12 .16 .21],'XColor',fg,'YColor',fg);hold(reviewAxLR,'on');grid(reviewAxLR,'on');
        reviewAxDelta=axes('Parent',reviewFig,'Units','normalized', ...
            'Position',[.075 .145 .87 .225], ...
            'Color',[.12 .16 .21],'XColor',fg,'YColor',fg);hold(reviewAxDelta,'on');grid(reviewAxDelta,'on');
        reviewText=uicontrol(reviewFig,'Style','text','Units','normalized', ...
            'Position',[.055 .018 .90 .105],'String','', ...
            'BackgroundColor',[.08 .105 .15],'ForegroundColor',fg, ...
            'FontName','Consolas','FontSize',10,'HorizontalAlignment','left');
        reviewCurLR=[];reviewCurD=[];
        updateReview();
    end

    function updateReview(varargin)
        if isempty(reviewFig)||~isgraphics(reviewFig)||isempty(reviewMenu) ...
        ||~isgraphics(reviewMenu)||nFrames==0,return;end
        pairNo=get(reviewMenu,'Value');pairNo=min(6,max(1,pairNo));
        hL=nan(nFrames,1);hR=hL;dH=hL;dV=hL;dD=hL;
        for kk=1:nFrames
            a=measurements(kk).pairs(pairNo);
            if a.valid
                hL(kk)=a.hL;hR(kk)=a.hR;dH(kk)=a.dH;
                dV(kk)=a.dV;dD(kk)=a.dD;
            end
        end
        cla(reviewAxLR);
        plot(reviewAxLR,frameNumbers,hL,'-o','Color',[.94 .65 .28], ...
            'LineWidth',1.7,'MarkerSize',3,'DisplayName','Left -> midline |U_L|');
        plot(reviewAxLR,frameNumbers,hR,'-o','Color',[.27 .80 .99], ...
            'LineWidth',1.7,'MarkerSize',3,'DisplayName','Right -> midline |U_R|');
        yl=smartLimit([hL;hR]);ylim(reviewAxLR,yl);xlim(reviewAxLR,smartX());
        reviewCurLR=plot(reviewAxLR,[frameNumbers(frame) frameNumbers(frame)],yl, ...
            '--','Color',[1 1 1],'HandleVisibility','off');
        legend(reviewAxLR,'Location','northeast');
        title(reviewAxLR,sprintf('%s: independent left/right distances (mm)',pairNames{pairNo}), ...
            'Color',fg,'Interpreter','none');
        xlabel(reviewAxLR,'Frame');ylabel(reviewAxLR,'Perpendicular distance (mm)');
        cla(reviewAxDelta);
        plot(reviewAxDelta,frameNumbers,dH,'-','Color',[1 .55 .25], ...
            'LineWidth',1.6,'DisplayName','dH = ||U_L|-|U_R||');
        plot(reviewAxDelta,frameNumbers,dV,'-','Color',[.36 .93 .51], ...
            'LineWidth',1.6,'DisplayName','dV = |V_L-V_R|');
        plot(reviewAxDelta,frameNumbers,dD,'-','Color',[.50 .64 1], ...
            'LineWidth',1.6,'DisplayName','dD = |depth_L-depth_R|');
        yd=smartLimit([dH;dV;dD]);ylim(reviewAxDelta,yd);xlim(reviewAxDelta,smartX());
        reviewCurD=plot(reviewAxDelta,[frameNumbers(frame) frameNumbers(frame)],yd, ...
            '--','Color',[1 1 1],'HandleVisibility','off');
        legend(reviewAxDelta,'Location','northeast');
        xlabel(reviewAxDelta,'Frame');ylabel(reviewAxDelta,'Pair difference (mm)');
        if isfinite(hL(frame))
            signed=hL(frame)-hR(frame);
            vals=sprintf(['Frame %d/%d | Left %.2f mm | Right %.2f mm | ' ...
                'L-R signed %.2f mm | dH %.2f | dV %.2f | dD %.2f mm'], ...
                frameNumbers(frame),frameNumbers(end),hL(frame),hR(frame), ...
                signed,dH(frame),dV(frame),dD(frame));
        else
            vals=sprintf('Frame %d: %s unavailable (missing, gated, or invalid point)', ...
                frameNumbers(frame),pairNames{pairNo});
        end
        set(reviewText,'String',{vals; ...
            '|U| is the shortest distance to U=0; dH ignores direction, but signed L-R retains direction.'});
    end

    function openLipQC(~,~)
        % Separation variability is NOT anatomical landmark validation.
        ls=idList({'LABIALE_SUPERIUS'});li=idList({'LABIALE_INFERIUS'});
        rawGap=nan(nFrames,1);smoothGap=nan(nFrames,1);seedGap=nan(nFrames,1);
        for ff=1:nFrames
            a=squeeze(rawSeeds(ff,ls,:));b=squeeze(rawSeeds(ff,li,:));
            if all(isfinite([a(:);b(:)])),seedGap(ff)=norm(a(:)-b(:));end
            a=squeeze(rawProcessed(ff,ls,:));b=squeeze(rawProcessed(ff,li,:));
            if all(isfinite([a(:);b(:)])),rawGap(ff)=norm(a(:)-b(:));end
            a=squeeze(smoothProcessed(ff,ls,:));b=squeeze(smoothProcessed(ff,li,:));
            if all(isfinite([a(:);b(:)])),smoothGap(ff)=norm(a(:)-b(:));end
        end
        f=figure('Name','LS / LI temporal-separation quality control', ...
            'NumberTitle','off','Color','w','Position',[180 120 1050 610]);
        a=axes('Parent',f,'Units','normalized','Position',[.085 .30 .87 .59]);
        hold(a,'on');grid(a,'on');
        plot(a,frameNumbers,seedGap,'-','Color',[.6 .6 .6], ...
            'LineWidth',1.6,'DisplayName','Original exported seeds');
        plot(a,frameNumbers,rawGap,'-o','Color',[.18 .45 .91], ...
            'MarkerSize',3,'DisplayName','Raw cloud-snapped');
        plot(a,frameNumbers,smoothGap,'-o','Color',[.10 .65 .40], ...
            'MarkerSize',3,'DisplayName','Causal 3-frame median');
        xlabel(a,'Frame');ylabel(a,'LS-to-LI Euclidean 3D separation (mm)');
        title(a,'LS/LI separation through time; 1 SD is a TEMPORAL statistic');
        legend(a,'Location','best');xlim(a,smartX());
        ok=isfinite(rawGap);sm=isfinite(smoothGap);
        summary=sprintf(['Raw cloud-snapped: mean %.3f mm | sample SD %s mm (n=%d)\n' ...
            'Median-smoothed: mean %.3f mm | sample SD %s mm (n=%d)\n' ...
            'Lower SD may indicate reduced jitter OR suppressed genuine lip motion. ' ...
            'Neither proves LS/LI are correctly placed.'], ...
            mean(rawGap(ok)),formatted(safeSD(rawGap)),sum(ok), ...
            mean(smoothGap(sm)),formatted(safeSD(smoothGap)),sum(sm));
        uicontrol(f,'Style','text','Units','normalized', ...
            'Position',[.07 .035 .90 .21],'String',summary, ...
            'HorizontalAlignment','left','BackgroundColor','w','FontSize',10);
    end

    function resetView(~,~)
        fitCloud();
        enableRotate();
    end

    function s=safeSD(v)
        v=v(isfinite(v));
        if numel(v)<2,s=NaN;else,s=std(v,0);end
    end

    function T=makeResults()
        if nFrames==0,T=table();return;end
        [speed,delta,units]=movementSeries();
        Frame=frameNumbers;TimeSeconds=acquiredTime;
        ValidLandmarks=zeros(nFrames,1);PairsUpper=ValidLandmarks;
        PairsLower=ValidLandmarks;
        U_H=nan(nFrames,1);U_V=U_H;U_D=U_H;UpperASI=U_H;
        L_H=U_H;L_V=U_H;L_D=U_H;LowerASI=U_H;
        O_H=U_H;O_V=U_H;O_D=U_H;OverallASI=U_H;
        for ii=1:nFrames
            m=measurements(ii);ValidLandmarks(ii)=m.validLandmarks;
            PairsUpper(ii)=m.upper.count;PairsLower(ii)=m.lower.count;
            U_H(ii)=m.upper.H;U_V(ii)=m.upper.V;U_D(ii)=m.upper.D;
            UpperASI(ii)=m.upper.ASI;
            L_H(ii)=m.lower.H;L_V(ii)=m.lower.V;L_D(ii)=m.lower.D;
            LowerASI(ii)=m.lower.ASI;
            O_H(ii)=m.overall.H;O_V(ii)=m.overall.V;O_D(ii)=m.overall.D;
            OverallASI(ii)=m.overall.ASI;
        end
        U_M=nan(nFrames,1);L_M=U_M;O_M=U_M;
        UpperBilateral=U_M;LowerBilateral=U_M;OverallBilateral=U_M;
        for ii=1:nFrames
            mm=measurements(ii);
            U_M(ii)=mm.upper.M;L_M(ii)=mm.lower.M;O_M(ii)=mm.overall.M;
            UpperBilateral(ii)=mm.upper.bilateralASI;
            LowerBilateral(ii)=mm.lower.bilateralASI;
            OverallBilateral(ii)=mm.overall.bilateralASI;
        end
        RunningUpperSD=nan(nFrames,1);RunningLowerSD=RunningUpperSD;
        RunningOverallSD=RunningUpperSD;
        for f=1:nFrames
            RunningUpperSD(f)=safeSD(UpperASI(1:f));
            RunningLowerSD(f)=safeSD(LowerASI(1:f));
            RunningOverallSD(f)=safeSD(OverallASI(1:f));
        end
        Speed=speed;VelocityChange=delta;MotionUnits=repmat({units},nFrames,1);
        LabelSource=frameSource;
        Selection=cell(nFrames,1);
        definition=sprintf('Pairs:%s | Midline:%s', ...
            strjoin(pairNames(selectedPairs),','), ...
            strjoin(IDs(selectedMid),','));
        Selection(:)={definition};
        T=table(Frame,TimeSeconds,ValidLandmarks,PairsUpper,PairsLower, ...
            U_H,U_V,U_D,U_M,UpperBilateral,UpperASI,RunningUpperSD, ...
            L_H,L_V,L_D,L_M,LowerBilateral,LowerASI,RunningLowerSD, ...
            O_H,O_V,O_D,O_M,OverallBilateral,OverallASI,RunningOverallSD, ...
            Speed,VelocityChange,MotionUnits,LabelSource,Selection);
    end

    function T=pairDetails()
        names={'Frame','Region','Pair','Left_ID','Right_ID','Selected','Valid', ...
            'hLeft_mm','hRight_mm','SignedLR_mm','Width_mm','dH_mm','dV_mm','dD_mm', ...
            'LX_mm','LY_mm','LZ_mm','RX_mm','RY_mm','RZ_mm', ...
            'MidX_mm','MidY_mm','MidZ_mm', ...
            'L_U_mm','L_V_mm','L_W_mm','R_U_mm','R_V_mm','R_W_mm', ...
            'Mid_U_mm','Mid_V_mm','Mid_W_mm'};
        rows=cell(nFrames*6,numel(names));
        at=0;
        for ii=1:nFrames
            for j=1:6
                at=at+1;t=measurements(ii).pairs(j);
                region='Upper';if j>=4,region='Lower';end
                rows(at,:)={frameNumbers(ii),region,t.label,t.L,t.R,selectedPairs(j),t.valid, ...
                    t.hL,t.hR,t.hL-t.hR,t.width,t.dH,t.dV,t.dD, ...
                    t.Lxyz(1),t.Lxyz(2),t.Lxyz(3), ...
                    t.Rxyz(1),t.Rxyz(2),t.Rxyz(3), ...
                    t.midxyz(1),t.midxyz(2),t.midxyz(3), ...
                    t.Luvw(1),t.Luvw(2),t.Luvw(3), ...
                    t.Ruvw(1),t.Ruvw(2),t.Ruvw(3), ...
                    t.miduvw(1),t.miduvw(2),t.miduvw(3)};
            end
        end
        T=cell2table(rows,'VariableNames',names);
    end

    function T=reviewDetails()
        at=0;rows=cell(nFrames*6,11);
        for ff=1:nFrames
            for j=1:6
                at=at+1;a=measurements(ff).pairs(j);
                rows(at,:)={frameNumbers(ff),pairNames{j},a.L,a.R, ...
                    a.valid,a.hL,a.hR,a.hL-a.hR,a.dH,a.dV,a.dD};
            end
        end
        T=cell2table(rows,'VariableNames', ...
            {'Frame','Pair','Left_ID','Right_ID','Valid', ...
            'LeftToMidline_mm','RightToMidline_mm','SignedLeftMinusRight_mm', ...
            'AbsHorizontalDifference_mm','VerticalDifference_mm','DepthDifference_mm'});
    end

    function T=midlineDetails()
        % Selected midline contributions. U offsets use the same frame's
        % nasion-to-subnasale reference, NOT an inferred contralateral point.
        slots=[upperMid(:)' lowerMid(:)'];
        names={'Frame','Region','Landmark','Selected','ReferenceAnchor', ...
            'Valid','U_mm','V_mm','W_mm','TwiceAbsU_mm', ...
            'RegionWidth_mm','M_contribution_pct'};
        rows=cell(nFrames*numel(slots),numel(names));at=0;
        for ff=1:nFrames
            q=squeeze(localXYZ(ff,:,:));m=measurements(ff);
            for j=slots
                at=at+1;
                reg='Upper';r=m.upper;
                if any(j==lowerMid),reg='Lower';r=m.lower;end
                anchor=j==idNasion||j==idSubnasale;
                ok=all(isfinite(q(j,:)))&&abs(q(j,3))<=65;
                use=selectedMid(j)&&ok&&~anchor;
                u=NaN;v=NaN;w=NaN;twice=NaN;contrib=NaN;
                if ok
                    u=q(j,1);v=q(j,2);w=q(j,3);
                    if anchor,twice=0;elseif use,twice=2*abs(u);end
                    if isfinite(twice)&&isfinite(r.sumW)&&r.sumW>0
                        contrib=100*twice/r.sumW;
                    end
                end
                rows(at,:)={frameNumbers(ff),reg,IDs{j},selectedMid(j), ...
                    anchor,ok,u,v,w,twice,r.sumW,contrib};
            end
        end
        T=cell2table(rows,'VariableNames',names);
    end

    function T=individualLandmarkDetails()
        % Every anatomical label gets its own 30-frame trace in mm;
        % no pairing/selection filter is applied to this audit export.
        nm={'Frame','Landmark_ID','Landmark_Name','Valid_Local', ...
            'Camera_X_mm','Camera_Y_mm','Camera_Z_mm', ...
            'U_mm','Abs_U_To_Midline_mm','V_mm','W_mm', ...
            'Step_3D_mm'};
        rows=cell(nFrames*nLM,numel(nm));at=0;
        for kk=1:nFrames
            for jj=1:nLM
                at=at+1;
                q=squeeze(localXYZ(kk,jj,:));
                p=squeeze(processed(kk,jj,:));
                d=NaN;
                if kk>1
                    prev=squeeze(processed(kk-1,jj,:));
                    if all(isfinite(prev))&&all(isfinite(p))
                        d=norm(p-prev);
                    end
                end
                rows(at,:)={frameNumbers(kk),IDs{jj},landmarkShortNames{jj}, ...
                    all(isfinite(q)),p(1),p(2),p(3),q(1),abs(q(1)), ...
                    q(2),q(3),d};
            end
        end
        T=cell2table(rows,'VariableNames',nm);
    end

    function T=selectedMeasurementDetails()
        d=selectedDistanceSeries();a=trailingAverage(d,movingWindow);
        RunningSD=nan(nFrames,1);RunningMean=RunningSD;
        for f=1:nFrames
            v=d(1:f);v=v(isfinite(v));
            if ~isempty(v),RunningMean(f)=mean(v);end
            RunningSD(f)=safeSD(v);
        end
        Frame=frameNumbers;Distance3D_mm=d;MovingMean_mm=a;
        Landmark_A=repmat(IDs(selectionA),nFrames,1);
        Landmark_B=repmat(IDs(selectionB),nFrames,1);
        T=table(Frame,Landmark_A,Landmark_B,Distance3D_mm,MovingMean_mm, ...
            RunningMean,RunningSD);
    end

    function showDetails(~,~)
        if nFrames==0,return;end
        T=pairDetails();T=T(T.Frame==frameNumbers(frame),:);
        f=figure('Name',sprintf('Pair components - frame %d',frameNumbers(frame)), ...
            'NumberTitle','off','Color','w','Position',[80 80 1480 380]);
        uitable(f,'Units','normalized','Position',[.02 .02 .96 .84], ...
            'Data',table2cell(T),'ColumnName',T.Properties.VariableNames);
        if any(selectedMid)
            Mt=midlineDetails();Mt=Mt(Mt.Frame==frameNumbers(frame),:);
            mf=figure('Name',sprintf('Selected midline offsets - frame %d', ...
                frameNumbers(frame)),'NumberTitle','off','Color','w', ...
                'Position',[120 140 1230 380]);
            uitable(mf,'Units','normalized','Position',[.02 .05 .96 .89], ...
                'Data',table2cell(Mt), ...
                'ColumnName',Mt.Properties.VariableNames);
        end
        uicontrol(f,'Style','text','Units','normalized','Position',[.02 .88 .95 .095], ...
            'String',['hL,hR = distances to interpolated midline in local U; ' ...
            'width=hL+hR; dH=|hL-hR|; dV=|V_L-V_R|; dD=|W_L-W_R|. ' ...
            'H/V/D=100*sum(delta)/sum(width). M=100*sum(2*|selected midline U|)/sum(width). ' ...
            'TOTAL=sqrt(H^2+V^2+D^2+M^2). Bilateral-only excludes M.'], ...
            'HorizontalAlignment','left','BackgroundColor','w');
    end

    function T=denseMeshDetails()
        % Every requested mesh ID appears once per frame, including NaNs.
        n=numel(denseIDs); data=cell(nFrames*n,5);at=0;
        for ff=1:nFrames
            for j=1:n
                at=at+1;
                p=squeeze(denseXYZ(ff,j,:));
                data(at,:)={frameNumbers(ff),denseIDs(j),p(1),p(2),p(3)};
            end
        end
        T=cell2table(data,'VariableNames', ...
            {'Frame','MediaPipe_Mesh_ID','X_mm','Y_mm','Z_mm'});
    end

    function T=eyelidQCDetails()
        [L,R]=eyelidSeries();
        LM=trailingAverage(L,movingWindow);RM=trailingAverage(R,movingWindow);
        T=table(frameNumbers,L,R,abs(L-R),LM,RM, ...
            repmat(movingWindow,nFrames,1), ...
            'VariableNames',{'Frame','Left_ApertureV_mm','Right_ApertureV_mm', ...
            'Absolute_Difference_mm','Left_MeanN_mm','Right_MeanN_mm','N'});
    end

    function exportTables(~,~)
        if nFrames==0,return;end
        [name,path]=uiputfile('*.csv','Save frame-by-frame ASI table','FacialASI_frames.csv');
        if isequal(name,0),return;end
        [~,stem,~]=fileparts(name);
        try
            T=makeResults();
            ASIraw=[arrayfun(@(m)m.upper.ASI,measurements(:)), ...
                arrayfun(@(m)m.lower.ASI,measurements(:)), ...
                arrayfun(@(m)m.overall.ASI,measurements(:))];
            T.Upper_Mean_N=trailingASI(ASIraw(:,1),'UPPER');
            T.Lower_Mean_N=trailingASI(ASIraw(:,2),'LOWER');
            T.Overall_Mean_N=trailingASI(ASIraw(:,3),'OVERALL');
            T.Moving_Window_N=repmat(movingWindow,nFrames,1);
            writetable(T,fullfile(path,name));
            writetable(pairDetails(),fullfile(path,[stem '_pair_details.csv']));
            writetable(midlineDetails(),fullfile(path,[stem '_midline_details.csv']));
            writetable(reviewDetails(),fullfile(path,[stem '_LR_comparison.csv']));
            writetable(individualLandmarkDetails(), ...
                fullfile(path,[stem '_all_individual_landmarks.csv']));
            writetable(selectedMeasurementDetails(), ...
                fullfile(path,[stem '_selected_AB_measurement.csv']));
            writetable(denseMeshDetails(), ...
                fullfile(path,[stem '_dense_eye_brow_nose.csv']));
            writetable(eyelidQCDetails(), ...
                fullfile(path,[stem '_eyelid_QC.csv']));
            assignin('base','FacialASI_Results',makeResults());
            assignin('base','FacialASI_PairDetails',pairDetails());
            msgbox('Saved ASI, all-landmark, and exact A-B measurements with running SD/moving mean.','Export complete');
        catch ME
            errordlg(ME.message,'Export failed');
        end
    end

    function openScanCSV(~,~)
        % Keep this recording open if the user cancels or import fails.
        MATLAB_FaceAssymmetry();
    end
    function showDeviceResults(varargin)
        if isempty(deviceFrames)
            msgbox('This source has no exported device scores. MATLAB panels show recalculated research values.');return;
        end
        f=figure('Name','Original device results (unchanged by MATLAB controls)','NumberTitle','off');
        names=deviceFrames.Properties.VariableNames;
        keep=contains(names,'asi_') | ismember(names,{'frame_index','elapsed_s','filter_window','filter_method', ...
            'expression_estimate','upper_filter_samples','lower_filter_samples','overall_filter_samples', ...
            'eye_gap_ratio_L','eye_gap_ratio_R','brow_gap_ratio_L','brow_gap_ratio_R'});
        Q=deviceFrames(:,keep);data=table2cell(Q);
        for rr=1:size(data,1),for cc=1:size(data,2)
                if isstring(data{rr,cc}),data{rr,cc}=char(data{rr,cc});end
            end,end
        uitable(f,'Data',data,'ColumnName',Q.Properties.VariableNames,'Units','normalized','Position',[.02 .02 .96 .78]);
        txt='Device scores preserved. Main viewer recalculates its own selected-landmark index; definitions may differ.';
        if ~isempty(deviceSummary)
            txt=sprintf('Whole scan device ASI: Overall %.2f%% | Upper %.2f%% | Lower %.2f%%. %d frames.\n%s', ...
                deviceSummary.scan_overall_mean_pct(1),deviceSummary.scan_upper_mean_pct(1), ...
                deviceSummary.scan_lower_mean_pct(1),height(deviceFrames),txt);
        end
        uicontrol(f,'Style','text','Units','normalized','Position',[.02 .82 .96 .16],'String',txt,'FontSize',12);
    end

    function closeViewer(~,~)
        playing=false;
        if exist('clockTimer','var')&&isvalid(clockTimer)
            if strcmp(clockTimer.Running,'on'),stop(clockTimer);end
            delete(clockTimer);
        end
        if ~isempty(reviewFig)&&isgraphics(reviewFig),delete(reviewFig);end
        if ~isempty(eyelidFig)&&isgraphics(eyelidFig),delete(eyelidFig);end
        if isgraphics(fig),delete(fig);end
    end
end

% === PUBLIC-RELEASE LANDMARK DICTIONARY / NO EMBEDDED HUMAN CAPTURE ===
% The original MATLAB capture was removed for participant privacy.
% Source CSV import, ASI geometry, plots, and all GUI callbacks are unchanged.

function D=embeddedData()
    % Public-release landmark dictionary; actual study scans are NOT included.
    D.landmarkIDs={'ALARE_L';'ALARE_R';'CHEILION_L';'CHEILION_R';'FOREHEAD_PROXY';'GLABELLA';'GNATHION';'LABIALE_INFERIUS';'LABIALE_SUPERIUS';'LATERAL_CANTHUS_L';'LATERAL_CANTHUS_R';'MEDIAL_CANTHUS_L';'MEDIAL_CANTHUS_R';'MENTON';'PRONASALE';'SELLION_NASION';'SUBNASALE';'SUPERCILIARE_L';'SUPERCILIARE_R';'ZYGOMA_L';'ZYGOMA_R'};
    D.labels={'Left Alare [ALARE_L]';'Right Alare [ALARE_R]';'Left Cheilion / Labial Commissure [CHEILION_L]';'Right Cheilion / Labial Commissure [CHEILION_R]';'Upper forehead (proxy) [FOREHEAD_PROXY]';'Glabella [GLABELLA]';'Gnathion / Chin [GNATHION]';'Labiale Inferius [LABIALE_INFERIUS]';'Labiale Superius [LABIALE_SUPERIUS]';'Left Lateral Canthus [LATERAL_CANTHUS_L]';'Right Lateral Canthus [LATERAL_CANTHUS_R]';'Left Medial Canthus [MEDIAL_CANTHUS_L]';'Right Medial Canthus [MEDIAL_CANTHUS_R]';'Menton [MENTON]';'Pronasale [PRONASALE]';'Sellion / Nasion [SELLION_NASION]';'Subnasale [SUBNASALE]';'Left Superciliare (proxy) [SUPERCILIARE_L]';'Right Superciliare (proxy) [SUPERCILIARE_R]';'Left Zygomatic Arch [ZYGOMA_L]';'Right Zygomatic Arch [ZYGOMA_R]'};
    D.xyz={};
end

function R=embeddedRawFrames() %#ok<STOUT>
    % The original version embeds one 30-frame facial recording (XYZ and RGB).
    % For public source distribution this recording is intentionally omitted.
    error('FaceASI:NoEmbeddedData', ...
        'No embedded face recording is included in the public release. Use ScanCSV with an authorized FaceASI scan export.');
end


function S=readScanCSV(filename,IDs)
    % One-file contract: landmark/cloud/dense_landmark/frame/scan_summary rows.
    opts=detectImportOptions(filename,'Delimiter',',','VariableNamingRule','preserve');
    required={'record_type','frame_index','landmark_id','x_mm','y_mm','z_mm'};
    if ~all(ismember(required,opts.VariableNames))
        error('Not a FaceASI scan CSV. Select the single Scan_###_Expression.csv export, not a per-frame internal file.');
    end
    textNames={'schema_version','scan_id','subject_id','mode','scan_key','record_type','landmark_id','filter_method', ...
        'summary_method','expression_estimate','metrics_json','quality_json','device_metadata_json','expression_json', ...
        'intrinsics_json','timestamp','upper_pair_subset','lower_pair_subset','overall_pair_subset'};
    textNames=intersect(textNames,opts.VariableNames,'stable');
    opts=setvartype(opts,textNames,'string');
    numericNames=setdiff(opts.VariableNames,textNames,'stable');
    opts=setvartype(opts,numericNames,'double');
    T=readtable(filename,opts);kind=T.record_type;
    if ~ismember('schema_version',T.Properties.VariableNames) || ...
            ~any(T.schema_version=="faceasi.scan.v1")
        error('Unsupported or missing FaceASI CSV schema version.');
    end
    lm=T(kind=="landmark",:);
    frames=unique(lm.frame_index);frames=frames(isfinite(frames));
    if isempty(frames)||any(frames<1|frames~=floor(frames)),error('No valid landmark frames in this CSV.');end
    n=numel(frames);S=struct();S.landmarkIDs=IDs;S.xyz=nan(n,numel(IDs),3);
    S.clouds=cell(n,1);S.timestamps=nan(n,1);S.frameNumbers=frames;
    d=T(kind=="dense_landmark",:);di=str2double(d.landmark_id);
    if any(~isfinite(di)|di<0|di>477|di~=floor(di)),error('Invalid dense mesh vertex IDs.');end
    S.denseIDs=unique(di)';S.denseXYZ=nan(n,numel(S.denseIDs),3);
    S.densePixels=nan(n,numel(S.denseIDs),2);S.deviceFrames=table();
    for k=1:n
        f=frames(k);r=lm(lm.frame_index==f,:);
        if numel(unique(r.landmark_id))~=height(r),error('Duplicate anatomical landmark within frame %d.',f);end
        [known,at]=ismember(r.landmark_id,string(IDs));
        xyz=[r.x_mm r.y_mm r.z_mm];xyz(~all(isfinite(xyz),2),:)=NaN;
        S.xyz(k,at(known),:)=reshape(xyz(known,:),1,sum(known),3);
        c=T(kind=="cloud" & T.frame_index==f,:);
        cloud=[c.x_mm c.y_mm c.z_mm];valid=all(isfinite(cloud),2);cloud=cloud(valid,:);
        if all(ismember({'r','g','b'},T.Properties.VariableNames))
            rgb=[c.r c.g c.b];rgb=rgb(valid,:);rgb(~isfinite(rgb))=160;
            cloud=[cloud max(0,min(255,rgb))];
        end
        S.clouds{k}=cloud;
        q=d(d.frame_index==f,:);[~,at]=ismember(str2double(q.landmark_id),S.denseIDs);
        if numel(unique(at))~=numel(at),error('Duplicate dense vertex within frame %d.',f);end
        if ~isempty(at)
            xyz=[q.x_mm q.y_mm q.z_mm];xyz(~all(isfinite(xyz),2),:)=NaN;
            S.denseXYZ(k,at,:)=reshape(xyz,1,numel(at),3);
            if all(ismember({'pixel_u','pixel_v'},T.Properties.VariableNames))
                S.densePixels(k,at,:)=reshape([q.pixel_u q.pixel_v],1,numel(at),2);
            end
        end
        fr=T(kind=="frame" & T.frame_index==f,:);
        if height(fr)>1,error('Duplicate frame record %d.',f);end
        if isempty(fr),fr=r(1,:);end % v53-v56 compatibility
        S.deviceFrames=[S.deviceFrames;fr]; %#ok<AGROW>
        if ismember('elapsed_s',fr.Properties.VariableNames),S.timestamps(k)=fr.elapsed_s(1);end
    end
    if all(isfinite(S.timestamps)) && n>1 && any(diff(S.timestamps)<=0)
        warning('FaceASI:Timing','Non-increasing timestamps: motion will use frame units.');
        S.timestamps(:)=NaN;
    end
    S.deviceSummary=T(kind=="scan_summary",:);
    if height(S.deviceSummary)>1,error('Expected one scan summary.');end
    S.sourceCSV=filename;
    % Store the exact source table, including JSON diagnostics; no data discarded.
    S.sourceTable=T;
end
