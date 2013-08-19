#ifndef ACTOR_H
#define ACTOR_H

#include "NNetwork.h"

namespace Neural
{
	class Actor
	{
		public:
			Actor();
			~Actor();

			void setNetwork(NNetwork* newNet);
			void clearNetwork();
			bool hasNetwork();

			//To be overriden by implementing classes
			virtual int getSensors() = 0;
			virtual int getActions() = 0;
			virtual float getFitness() = 0;
			virtual void reset() = 0;

		protected:
			NNetwork* network;
	};
}

#endif
