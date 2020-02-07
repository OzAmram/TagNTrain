#include "NsubjettinessWrapper.h"
#include "fastjet/PseudoJet.hh"
//stolen from here https://github.com/cms-jet/NanoAODJMARTools/blob/master/src/NsubjettinessWrapper.cc

Nsubjettiness::Nsubjettiness(double beta, double R0, double Rcutoff, unsigned measureDefinition, unsigned axesDefinition, int nPass, double akAxesR0 ) :
    beta_(beta), R0_(R0), 
    Rcutoff_(Rcutoff), measureDefinition_(measureDefinition),
    axesDefinition_(axesDefinition),
    nPass_(nPass), akAxesR0_(akAxesR0) 
{
    // Get the measure definition
    fastjet::contrib::NormalizedMeasure          normalizedMeasure        (beta_,R0_);
    fastjet::contrib::UnnormalizedMeasure        unnormalizedMeasure      (beta_);
    fastjet::contrib::OriginalGeometricMeasure   geometricMeasure         (beta_);// changed in 1.020
    fastjet::contrib::NormalizedCutoffMeasure    normalizedCutoffMeasure  (beta_,R0_,Rcutoff_);
    fastjet::contrib::UnnormalizedCutoffMeasure  unnormalizedCutoffMeasure(beta_,Rcutoff_);

    fastjet::contrib::MeasureDefinition const * measureDef = nullptr;
    switch ( measureDefinition_ ) {
        case UnnormalizedMeasure : measureDef = &unnormalizedMeasure; break;
        case OriginalGeometricMeasure    : measureDef = &geometricMeasure; break;// changed in 1.020
        case NormalizedCutoffMeasure : measureDef = &normalizedCutoffMeasure; break;
        case UnnormalizedCutoffMeasure : measureDef = &unnormalizedCutoffMeasure; break;
        case NormalizedMeasure : default : measureDef = &normalizedMeasure; break;
    }

    // Get the axes definition
    fastjet::contrib::KT_Axes             kt_axes;
    fastjet::contrib::CA_Axes             ca_axes;
    fastjet::contrib::AntiKT_Axes         antikt_axes   (akAxesR0_);
    fastjet::contrib::WTA_KT_Axes         wta_kt_axes;
    fastjet::contrib::WTA_CA_Axes         wta_ca_axes;
    fastjet::contrib::OnePass_KT_Axes     onepass_kt_axes;
    fastjet::contrib::OnePass_CA_Axes     onepass_ca_axes;
    fastjet::contrib::OnePass_AntiKT_Axes onepass_antikt_axes   (akAxesR0_);
    fastjet::contrib::OnePass_WTA_KT_Axes onepass_wta_kt_axes;
    fastjet::contrib::OnePass_WTA_CA_Axes onepass_wta_ca_axes;
    fastjet::contrib::MultiPass_Axes      multipass_axes (nPass_);

    fastjet::contrib::AxesDefinition const * axesDef = nullptr;
    switch ( axesDefinition_ ) {
        case  KT_Axes : default : axesDef = &kt_axes; break;
        case  CA_Axes : axesDef = &ca_axes; break;
        case  AntiKT_Axes : axesDef = &antikt_axes; break;
        case  WTA_KT_Axes : axesDef = &wta_kt_axes; break;
        case  WTA_CA_Axes : axesDef = &wta_ca_axes; break;
        case  OnePass_KT_Axes : axesDef = &onepass_kt_axes; break;
        case  OnePass_CA_Axes : axesDef = &onepass_ca_axes; break;
        case  OnePass_AntiKT_Axes : axesDef = &onepass_antikt_axes; break;
        case  OnePass_WTA_KT_Axes : axesDef = &onepass_wta_kt_axes; break;
        case  OnePass_WTA_CA_Axes : axesDef = &onepass_wta_ca_axes; break;
        case  MultiPass_Axes : axesDef = &multipass_axes; break;
    };

    routine_ = std::auto_ptr<fastjet::contrib::Njettiness> ( new fastjet::contrib::Njettiness( *axesDef, *measureDef ) );
}

Nsubjettiness::Nsubjettiness(double beta, double R0, unsigned measureDefinition, unsigned axesDefinition ) :
    Nsubjettiness::Nsubjettiness( beta, R0, 999., measureDefinition, axesDefinition, 999., 999. )
{}


 
Nsubjettiness::Nsubjettiness() :
    beta_(1), R0_(0.8), 
    Rcutoff_(999.), measureDefinition_(0),
    axesDefinition_(6),
    nPass_(999), akAxesR0_(999) 
{
}

std::vector<double> Nsubjettiness::getTau( unsigned maxTau, std::vector<double> particles)
//std::vector<double> Nsubjettiness::getTau( unsigned maxTau, const std::vector<fastjet::PseudoJet> pjets)
{
    fastjet::JetDefinition jet_def (fastjet::antikt_algorithm, R0_);
    std::vector<fastjet::PseudoJet> pjets;
    int width = 4;
    int nParticles = particles.size()/width;
    for( int i=0; i<nParticles; i++) {
        pjets.emplace_back(particles[i*width], particles[i*width+1], particles[i*width+2], particles[i*width+3]);
    }


    std::vector<double> vecTauN;
    for( unsigned tau = 1; tau <= maxTau; tau++ ) {
        double t = routine_->getTau(tau, pjets); 
        vecTauN.push_back(t);
    }
    return vecTauN;
}
