#ifndef NNETWORK_H
#define NNETWORK_H

#include "NWeight.h"
#include <map>
#include <vector>
#include <list>

namespace Neural 
{
	class Actor;

	class NNetwork
	{
		public:
			NNetwork(const std::vector<NWeight> *weights, int inputs, int outputs, const std::vector<int> *hiddenNodes);
			NNetwork(const NNetwork* n);
			~NNetwork();

			float getOutput(int output);
			void setInput(int input, float value);

			void step();
			void reset();

			void setActor(Actor* actor) { this->actor = actor; }
			Actor* getActor() const { return actor; }

			int numInputs() const { return inputs; }
			int numOutputs() const { return outputs; }
			int numWeights() const { return numweights; }
			const std::vector<NWeight> *getWeights() const { return weights; }
			const std::vector<int> *getHidden() const { return hiddenNodes; }
			bool hasConnection(int nodeA, int nodeB);
			void printNetwork();

			static float calculateDelta(const NNetwork* a, const NNetwork* b, float c1, float c3);

		private:
			std::map<int, float> *neurodes;
			std::map<int, float> *neurodeBuffer;
			std::vector<int> *hiddenNodes;

			std::vector<NWeight> *weights;
			int numweights;
			int inputs;
			int outputs;
			Actor* actor;
	};

}

#endif
