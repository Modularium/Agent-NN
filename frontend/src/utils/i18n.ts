import { useState, useEffect, createContext, useContext } from 'react';
import deTranslations from '../translations/de.json';
import enTranslations from '../translations/en.json';

// Available translations
const translations = {
  de: deTranslations,
  en: enTranslations
};

// Types
type Language = 'de' | 'en';
type TranslationKey = string;
type TranslationParams = Record<string, string | number>;

// Context
interface I18nContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: TranslationKey, params?: TranslationParams) => string;
  availableLanguages: Language[];
}

const I18nContext = createContext<I18nContextType>({
  language: 'de',
  setLanguage: () => {},
  t: (key) => key,
  availableLanguages: ['de', 'en']
});

// Provider
interface I18nProviderProps {
  children: React.ReactNode;
  defaultLanguage?: Language;
}

export const I18nProvider: React.FC<I18nProviderProps> = ({ 
  children, 
  defaultLanguage = 'de' 
}) => {
  const [language, setLanguage] = useState<Language>(defaultLanguage);

  // Load language from localStorage on mount
  useEffect(() => {
    const savedLanguage = localStorage.getItem('language') as Language;
    if (savedLanguage && translations[savedLanguage]) {
      setLanguage(savedLanguage);
    }
  }, []);

  // Save language to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('language', language);
  }, [language]);

  // Translation function
  const t = (key: TranslationKey, params?: TranslationParams): string => {
    // Split the key by dots to access nested properties
    const keys = key.split('.');
    
    // Get the translation object for the current language
    let translation: any = translations[language];
    
    // Navigate through the nested properties
    for (const k of keys) {
      if (!translation || !translation[k]) {
        // If the key doesn't exist, return the key itself
        return key;
      }
      translation = translation[k];
    }
    
    // If the translation is not a string, return the key
    if (typeof translation !== 'string') {
      return key;
    }
    
    // Replace parameters if provided
    if (params) {
      return Object.entries(params).reduce((acc, [paramKey, paramValue]) => {
        return acc.replace(new RegExp(`{{${paramKey}}}`, 'g'), String(paramValue));
      }, translation);
    }
    
    return translation;
  };

  return (
    <I18nContext.Provider value={{ 
      language, 
      setLanguage, 
      t,
      availableLanguages: ['de', 'en']
    }}>
      {children}
    </I18nContext.Provider>
  );
};

// Hook
export const useTranslation = () => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useTranslation must be used within an I18nProvider');
  }
  return context.t;
};

// Hook to get and set language
export const useLanguage = () => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useLanguage must be used within an I18nProvider');
  }
  return {
    language: context.language,
    setLanguage: context.setLanguage,
    availableLanguages: context.availableLanguages
  };
};

export default {
  I18nProvider,
  useTranslation,
  useLanguage
};