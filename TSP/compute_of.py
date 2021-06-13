from math import atan2
import pyflow


def compute_of(oim0, oim1, outname):

# % add the optical flow path
# addpath('optical_flow_celiu/');
# addpath('optical_flow_celiu/mex/');

    overwrite = false;

    if (overwrite || ~exist(outname, 'file'))
        [~, flow.bvx, flow.bvy] = compute_flow(oim1, oim0);
        [~, flow.fvx, flow.fvy] = compute_flow(oim0, oim1);
        save(outname, 'flow');
    end


def compute_flow(im1, im2):

    if max(im1) > 1:
        im1 = float(im1) / 255
    if max(im2) > 1:
        im2 = float(im2) / 255

    # % set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
    alpha = 0.012
    # %alpha = 0.05
    ratio = 0.75
    # %ratio = 1
    minWidth = 50
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0

    para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations]

    # % this is the core part of calling the mexed dll file for computing optical flow
    # % it also returns the time that is needed for two-frame estimation
    vx, vy, warpI2 = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    angle = atan2(vy, vx);
    return angle, vx, vy