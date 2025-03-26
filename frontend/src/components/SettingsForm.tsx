import React, { useState } from 'react';
import { 
  Card, 
  Input, 
  Button, 
  Select,
  Switch,
  Alert,
  useTranslation
} from '@smolitux/core';
import { useLanguage } from '../utils/i18n';
import { useTheme } from '../utils/theme';
import { Settings } from '../types';

interface SettingsFormProps {
  initialSettings: Settings;
  onSave: (settings: Settings) => void;
}

const SettingsForm: React.FC<SettingsFormProps> = ({ 
  initialSettings, 
  onSave 
}) => {
  const [settings, setSettings] = useState<Settings>(initialSettings);
  const [saved, setSaved] = useState(false);
  const t = useTranslation();
  const { language, setLanguage, availableLanguages } = useLanguage();
  const { theme, setTheme } = useTheme();

  const handleChange = (field: keyof Settings, value: any) => {
    setSettings({
      ...settings,
      [field]: value
    });
    setSaved(false);
  };

  const handleSave = () => {
    onSave(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
    
    // Update language if changed
    if (settings.language !== language) {
      setLanguage(settings.language as any);
    }
  };

  const llmOptions = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'lmstudio', label: 'LM Studio' },
    { value: 'local', label: 'Local Model' }
  ];

  const languageOptions = availableLanguages.map(lang => ({
    value: lang,
    label: lang === 'de' ? 'Deutsch' : 'English'
  }));

  return (
    <div className="settings-form">
      {saved && (
        <Alert type="success" className="mb-2">
          {t('settings.saved')}
        </Alert>
      )}
      
      <Card className="settings-section">
        <h2>{t('settings.llmSettings')}</h2>
        
        <div className="settings-group">
          <label>{t('settings.llmBackend')}</label>
          <Select 
            options={llmOptions} 
            value={settings.llmBackend}
            onChange={(value) => handleChange('llmBackend', value)}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.apiKey')}</label>
          <Input 
            type="password"
            value={settings.apiKey}
            onChange={(e) => handleChange('apiKey', e.target.value)}
            placeholder={t('settings.apiKeyPlaceholder')}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.maxTokens')}</label>
          <Input 
            type="number"
            value={settings.maxTokens.toString()}
            onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
            min="100"
            max="4000"
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.temperature')}</label>
          <Input 
            type="range"
            value={settings.temperature.toString()}
            onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
            min="0"
            max="1"
            step="0.1"
          />
          <span>{settings.temperature}</span>
        </div>
        
        <div className="settings-group">
          <label>{t('settings.enableLocalModels')}</label>
          <Switch 
            checked={settings.enableLocalModels}
            onChange={() => handleChange('enableLocalModels', !settings.enableLocalModels)}
          />
        </div>
      </Card>
      
      <Card className="settings-section">
        <h2>{t('settings.systemSettings')}</h2>
        
        <div className="settings-group">
          <label>{t('settings.enableLogging')}</label>
          <Switch 
            checked={settings.enableLogging}
            onChange={() => handleChange('enableLogging', !settings.enableLogging)}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.language')}</label>
          <Select 
            options={languageOptions} 
            value={settings.language}
            onChange={(value) => handleChange('language', value)}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('theme.light')}/{t('theme.dark')}</label>
          <Switch 
            checked={theme === 'dark'}
            onChange={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          />
        </div>
      </Card>
      
      <div className="settings-actions">
        <Button 
          variant="primary" 
          onClick={handleSave}
        >
          {t('settings.save')}
        </Button>
        <Button 
          variant="secondary"
          onClick={() => setSettings(initialSettings)}
        >
          {t('settings.cancel')}
        </Button>
      </div>
    </div>
  );
};

export default SettingsForm;