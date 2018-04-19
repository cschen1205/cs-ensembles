using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Lang;
using Solvers;

namespace Ensemble
{
    public class WeakClassifier<T, U> : Classifier<T, U>
        where T : DataRecord<U>
    {
        public virtual double WeakPredict(T rec)
        {
            return Predict(rec) == rec.Label ? 1 : -1;
        }
    }
}
