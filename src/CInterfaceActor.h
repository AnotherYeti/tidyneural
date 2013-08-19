#ifndef CACTORINTERFACE_H
#define CACTORINTERFACE_H

#include "Actor.h"

extern "C" {
	void* buildActor(float (*fitnessFunc)(), int sensors, int actions, void (*resetFunc)(), void (*destroyFunc)());
	void destroyPopulation(void* in);
	void setInput(void* actor, int input, float value);
	float getOutput(void* actor, int output);
	bool hasNetwork(void* actor);
	void stepNetwork(void* actor);

	void* buildPopulation(void** actors, int size);
	void setWeightMutationRate(void* population, float newRate);
	void setWeightMutationIntensity(void* population, float newRate);
	void setWeightDisableRate(void* population, float newRate);
	void setAddWeightRate(void* population, float newRate);
	void setAddNodeRate(void* population, float newRate);
	void setc1(void* population, float c1);
	void setc3(void* population, float c3);
	void setSpeciationDifference(void* population, float newSpeciation);

	void stepPopulationGeneration(void* population);
}

namespace Neural
{
	class CActor : public Neural::Actor
	{
		public:
			CActor(float (fitnessFunc()), int sensors, int actions, void resetFunc(), void destroyFunc());
			~CActor();

			int getSensors() { return sensors; }
			int getActions() { return actions; }
			NNetwork* getNetwork() { return network; }
			float getFitness();
			void reset();

		private:
			float (*fitnessFunc)();
			void (*resetFunc)();
			void (*destroyFunc)();
			int sensors;
			int actions;
	};
}

#endif
