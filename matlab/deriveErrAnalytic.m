function [ Jac, residual ] = deriveErrAnalytic( IRef, DRef, I, xi, K )
    % calculate analytic derivative

    % get shorthands (R, t)
    T = se3Exp(xi);
    R = T(1:3, 1:3);
    t = T(1:3,4);

    % ========= warp pixels into other image, save intermediate results ===============
    % these contain the x,y image coordinates of the respective
    % reference-pixel, transformed & projected into the new image.
    xImg = zeros(size(IRef))-10;
    yImg = zeros(size(IRef))-10;

    % these contain the 3d position of the transformed point
    xp = NaN(size(IRef));
    yp = NaN(size(IRef));
    zp = NaN(size(IRef));
    for x=1:size(IRef,2)
        for y=1:size(IRef,1)
            % TODO warp points into target frame
        end
    end

    % ========= calculate actual derivative. ===============
    % calculate image derivatives 
    % TODO image gradient in x and y direction using central differences
    %dxI = ...
    %dyI = ...
    % interpolate at warped positions
    dxInterp = K(1,1) * reshape(interp2(dxI, xImg+1, yImg+1),size(I,1) * size(I,2),1);
    dyInterp = K(2,2) * reshape(interp2(dyI, xImg+1, yImg+1),size(I,1) * size(I,2),1);

    % 2.: get warped 3d points (x', y', z').
    xp = reshape(xp,size(I,1) * size(I,2),1);
    yp = reshape(yp,size(I,1) * size(I,2),1);
    zp = reshape(zp,size(I,1) * size(I,2),1);

    % 3. direct implementation of kerl2012msc.pdf Eq. (4.14):
    Jac = zeros(size(I,1) * size(I,2),6);
    % TODO implement analytic partial derivatives
    %Jac(:,1) = ...
    %Jac(:,2) = ...
    %Jac(:,3) = ...
    %Jac(:,4) = ...
    %Jac(:,5) = ...
    %Jac(:,6) = ...
    % invert jacobian: in kerl2012msc.pdf, the difference is defined the other
    % way round, see (4.6).
    Jac = -Jac;

    % ========= plot residual image =========
    residual = reshape(IRef - interp2(I, xImg+1, yImg+1),size(I,1) * size(I,2),1);
    imagesc(reshape(residual,size(I)));
    colormap gray;
    set(gca, 'CLim', [-1,1]);
end

