// monitoring/dashboard/components/common/FormWizard.tsx
import React, { useState, useEffect, ReactNode } from 'react';
import { Check, ArrowRight, ArrowLeft } from 'lucide-react';

export interface Step {
  id: string;
  title: string;
  description?: string;
  component: ReactNode;
  validator?: () => boolean | Promise<boolean>;
}

export interface FormWizardProps {
  steps: Step[];
  onComplete: (data: any) => void;
  onCancel: () => void;
  initialData?: any;
  className?: string;
  submitButtonText?: string;
  cancelButtonText?: string;
  backButtonText?: string;
  nextButtonText?: string;
  showProgressBar?: boolean;
  showStepNumbers?: boolean;
}

const FormWizard: React.FC<FormWizardProps> = ({
  steps,
  onComplete,
  onCancel,
  initialData = {},
  className = '',
  submitButtonText = 'Submit',
  cancelButtonText = 'Cancel',
  backButtonText = 'Back',
  nextButtonText = 'Next',
  showProgressBar = true,
  showStepNumbers = true,
}) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [stepData, setStepData] = useState<Record<string, any>>(initialData);
  const [isValidating, setIsValidating] = useState(false);
  const [stepStatus, setStepStatus] = useState<('incomplete' | 'complete' | 'current')[]>(
    steps.map((_, index) => (index === 0 ? 'current' : 'incomplete'))
  );

  // Handle data changes
  const updateStepData = (stepId: string, data: any) => {
    setStepData((prevData) => ({
      ...prevData,
      [stepId]: data,
    }));
  };

  // Validate current step
  const validateCurrentStep = async (): Promise<boolean> => {
    const currentStep = steps[currentStepIndex];
    
    if (!currentStep.validator) {
      return true; // No validator means step is always valid
    }

    setIsValidating(true);
    try {
      const isValid = await currentStep.validator();
      return isValid;
    } catch (error) {
      console.error('Error validating step:', error);
      return false;
    } finally {
      setIsValidating(false);
    }
  };

  // Navigate to next step
  const handleNext = async () => {
    const isValid = await validateCurrentStep();
    if (!isValid) return;

    if (currentStepIndex < steps.length - 1) {
      // Update step status
      const newStatus = [...stepStatus];
      newStatus[currentStepIndex] = 'complete';
      newStatus[currentStepIndex + 1] = 'current';
      setStepStatus(newStatus);
      
      // Move to next step
      setCurrentStepIndex(currentStepIndex + 1);
    }
  };

  // Navigate to previous step
  const handleBack = () => {
    if (currentStepIndex > 0) {
      // Update step status
      const newStatus = [...stepStatus];
      newStatus[currentStepIndex] = 'incomplete';
      newStatus[currentStepIndex - 1] = 'current';
      setStepStatus(newStatus);
      
      // Move to previous step
      setCurrentStepIndex(currentStepIndex - 1);
    }
  };

  // Handle form submission
  const handleSubmit = async () => {
    const isValid = await validateCurrentStep();
    if (!isValid) return;

    onComplete(stepData);
  };

  // If the steps change (e.g. dynamically loaded), reset the status
  useEffect(() => {
    setStepStatus(steps.map((_, index) => (index === currentStepIndex ? 'current' : index < currentStepIndex ? 'complete' : 'incomplete')));
  }, [steps, currentStepIndex]);

  const currentStep = steps[currentStepIndex];
  const isLastStep = currentStepIndex === steps.length - 1;

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow ${className}`}>
      {/* Header with steps */}
      <div className="border-b border-gray-200 dark:border-gray-700 p-4">
        <nav aria-label="Progress">
          <ol className="flex items-center">
            {steps.map((step, index) => (
              <li 
                key={step.id} 
                className={`relative flex-1 ${index > 0 ? 'ml-6' : ''}`}
              >
                {index > 0 && (
                  <div 
                    className="absolute inset-0 flex items-center" 
                    aria-hidden="true"
                  >
                    <div 
                      className={`h-0.5 w-full ${
                        stepStatus[index] === 'complete' || stepStatus[index - 1] === 'complete'
                          ? 'bg-indigo-600 dark:bg-indigo-500'
                          : 'bg-gray-200 dark:bg-gray-700'
                      }`}
                    ></div>
                  </div>
                )}
                <div 
                  className={`relative flex items-center justify-center ${
                    index !== steps.length - 1 ? 'group' : ''
                  }`}
                >
                  {stepStatus[index] === 'complete' ? (
                    <span className="h-8 w-8 rounded-full bg-indigo-600 dark:bg-indigo-500 flex items-center justify-center">
                      <Check className="w-5 h-5 text-white" />
                      <span className="sr-only">Complete</span>
                    </span>
                  ) : stepStatus[index] === 'current' ? (
                    <span className="h-8 w-8 rounded-full border-2 border-indigo-600 dark:border-indigo-500 flex items-center justify-center bg-white dark:bg-gray-800">
                      {showStepNumbers && <span className="text-indigo-600 dark:text-indigo-500">{index + 1}</span>}
                      <span className="sr-only">Step {index + 1}</span>
                    </span>
                  ) : (
                    <span className="h-8 w-8 rounded-full border-2 border-gray-300 dark:border-gray-600 flex items-center justify-center bg-white dark:bg-gray-800">
                      {showStepNumbers && <span className="text-gray-500 dark:text-gray-400">{index + 1}</span>}
                      <span className="sr-only">Step {index + 1}</span>
                    </span>
                  )}
                  <span className={`ml-2 text-sm font-medium ${
                    stepStatus[index] === 'current' 
                      ? 'text-indigo-600 dark:text-indigo-400' 
                      : stepStatus[index] === 'complete'
                        ? 'text-indigo-600 dark:text-indigo-400'
                        : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {step.title}
                  </span>
                </div>
              </li>
            ))}
          </ol>
        </nav>

        {/* Progress bar */}
        {showProgressBar && (
          <div className="mt-4">
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-indigo-600 dark:bg-indigo-500 rounded-full"
                style={{ width: `${((currentStepIndex + 1) / steps.length) * 100}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>Step {currentStepIndex + 1} of {steps.length}</span>
              <span>{Math.round(((currentStepIndex + 1) / steps.length) * 100)}% Complete</span>
            </div>
          </div>
        )}
      </div>

      {/* Step content */}
      <div className="p-6">
        <div className="mb-4">
          <h2 className="text-lg font-medium text-gray-900 dark:text-white">
            {currentStep.title}
          </h2>
          {currentStep.description && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {currentStep.description}
            </p>
          )}
        </div>

        {/* Render the component for the current step */}
        <div>
          {React.isValidElement(currentStep.component)
            ? React.cloneElement(currentStep.component as React.ReactElement, {
                stepData: stepData[currentStep.id] || {},
                updateStepData: (data: any) => updateStepData(currentStep.id, data),
                allData: stepData,
              })
            : currentStep.component}
        </div>
      </div>

      {/* Footer with navigation buttons */}
      <div className="border-t border-gray-200 dark:border-gray-700 p-4 flex justify-between">
        <div>
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800"
          >
            {cancelButtonText}
          </button>
        </div>
        <div className="flex space-x-3">
          {currentStepIndex > 0 && (
            <button
              type="button"
              onClick={handleBack}
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-indigo-700 dark:text-indigo-300 bg-indigo-100 dark:bg-indigo-900/30 border border-transparent rounded-md hover:bg-indigo-200 dark:hover:bg-indigo-900/50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800"
            >
              <ArrowLeft className="w-4 h-4 mr-1" />
              {backButtonText}
            </button>
          )}
          <button
            type="button"
            onClick={isLastStep ? handleSubmit : handleNext}
            disabled={isValidating}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-indigo-600 dark:bg-indigo-700 border border-transparent rounded-md hover:bg-indigo-700 dark:hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800 disabled:opacity-70"
          >
            {isValidating ? (
              <>
                <span className="animate-spin rounded-full h-4 w-4 border-t-2 border-white mr-1"></span>
                Validating...
              </>
            ) : isLastStep ? (
              submitButtonText
            ) : (
              <>
                {nextButtonText}
                <ArrowRight className="w-4 h-4 ml-1" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FormWizard;
