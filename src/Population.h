#ifndef POPULATION_H
#define POPULATION_H

#include "NNetwork.h"
#include "Actor.h"
#include <list>

namespace NEAT
{
	class Population 
	{
		public:
			Population(std::list<Neural::Actor*> *actors);
			~Population();

			void stepGeneration();
			void stepGeneration(bool verbose);

			//Variables
			float weightMutationRate;
			float weightMutationIntensity;
			float weightDisableRate;
			float addWeightMutationRate;
			float addNodeMutationRate;
			float c1;
			float c3;
			float speciationDifference;

			void printGeneration();

			std::list<Neural::Actor*> *getActors() { return actors; }

		private:
			Neural::NNetwork* mutateWeights(bool verbose, Neural::NNetwork* net);
			double randRange(double lower, double upper);
			static bool compare_fitness(Neural::NNetwork* first, Neural::NNetwork* second);

			std::list<Neural::NNetwork*> *nets;
			std::list<Neural::Actor*> *actors;

			std::map<int, int> node_innovations;
			std::map<int, std::pair<int, int> > new_weights_node_innovations;
			std::map<std::pair<int, int>, int> weight_innovations;

			int weight_innovation;
			int node_innovation;
	};
}

#endif
