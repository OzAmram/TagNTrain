//wrapper for fastjet-contrib nsubjettiness function
//shamelessly stolen from https://github.com/cms-jet/NanoAODJMARTools/blob/master/src/NsubjettinessWrapper.cc


#ifndef PhysicsTools_NanoAODToolsJMAR_NsubjettinessWrapper_h
#define PhysicsTools_NanoAODToolsJMAR_NsubjettinessWrapper_h

#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/Njettiness.hh"
#include "fastjet/contrib/NjettinessPlugin.hh"

#include <string>
#include <vector>

class Nsubjettiness {
    
    public:
        Nsubjettiness(double beta, double R0, double Rcutoff, unsigned measureDefinition, unsigned axesDefinition, int nPass, double akAxesR0 );
        Nsubjettiness(double beta, double R0, unsigned measureDefinition, unsigned axesDefinition );
        Nsubjettiness();

        std::vector<double> getTau( unsigned maxTau, std::vector<double> particles);
        //std::vector<double> getTau( unsigned maxTau, const std::vector<fastjet::PseudoJet> pjets);

        enum MeasureDefinition_t {
            NormalizedMeasure=0,       // (beta,R0) 
            UnnormalizedMeasure,       // (beta) 
            OriginalGeometricMeasure,  // (beta) 
            NormalizedCutoffMeasure,   // (beta,R0,Rcutoff) 
            UnnormalizedCutoffMeasure, // (beta,Rcutoff) 
            GeometricCutoffMeasure,    // (beta,Rcutoff) 
            N_MEASURE_DEFINITIONS
        };

        enum AxesDefinition_t {
            KT_Axes=0,
            CA_Axes,
            AntiKT_Axes,   // (axAxesR0)
            WTA_KT_Axes,
            WTA_CA_Axes,
            Manual_Axes,
            OnePass_KT_Axes,
            OnePass_CA_Axes,
            OnePass_AntiKT_Axes,   // (axAxesR0)
            OnePass_WTA_KT_Axes,
            OnePass_WTA_CA_Axes,
            OnePass_Manual_Axes,
            MultiPass_Axes,
            N_AXES_DEFINITIONS
        };

    private:

        // Measure definition :
        double          beta_ ;
        double          R0_;
        double          Rcutoff_;
        unsigned        measureDefinition_;

        // Axes definition :
        unsigned        axesDefinition_;
        int             nPass_;
        double          akAxesR0_;

        std::auto_ptr<fastjet::contrib::Njettiness> routine_;
};

#endif
