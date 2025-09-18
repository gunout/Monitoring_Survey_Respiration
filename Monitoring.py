import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import sqlite3
import json

class AdvancedPatientMonitor:
    def __init__(self, patient_id, patient_name, config_file="config.json"):
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.config = self.load_config(config_file)
        self.data = {
            'timestamp': [],
            'spo2': [],
            'heart_rate': [],
            'respiratory_rate': [],
            'temperature': [],
            'systolic_bp': [],
            'diastolic_bp': [],
            'symptoms': [],
            'activity_level': []
        }
        self.symptom_log = []
        self.alerts = []
        self.setup_database()
        
    def load_config(self, config_file):
        """Charge la configuration à partir d'un fichier JSON"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Configuration par défaut
            return {
                "alert_thresholds": {
                    "spo2_low": 92,
                    "heart_rate_high": 110,
                    "respiratory_rate_high": 25,
                    "temperature_high": 38.0
                },
                "email_alerts": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipient_emails": []
                },
                "prediction_settings": {
                    "window_size": 10,
                    "forecast_hours": 6
                }
            }
    
    def setup_database(self):
        """Configure la base de données SQLite"""
        self.conn = sqlite3.connect(f'patient_{self.patient_id}.db')
        self.cursor = self.conn.cursor()
        
        # Création des tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                spo2 REAL,
                heart_rate REAL,
                respiratory_rate REAL,
                temperature REAL,
                systolic_bp REAL,
                diastolic_bp REAL,
                activity_level TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS symptoms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symptom TEXT,
                severity INTEGER,
                notes TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                alert_type TEXT,
                message TEXT,
                severity TEXT
            )
        ''')
        
        self.conn.commit()
    
    def save_measurement(self, measurement):
        """Enregistre une mesure dans la base de données"""
        self.cursor.execute('''
            INSERT INTO measurements 
            (timestamp, spo2, heart_rate, respiratory_rate, temperature, systolic_bp, diastolic_bp, activity_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            measurement['time'],
            measurement['spo2'],
            measurement['heart_rate'],
            measurement['respiratory_rate'],
            measurement.get('temperature', 0),
            measurement.get('systolic_bp', 0),
            measurement.get('diastolic_bp', 0),
            measurement['activity_level']
        ))
        self.conn.commit()
    
    def save_symptom(self, symptom_entry):
        """Enregistre un symptôme dans la base de données"""
        self.cursor.execute('''
            INSERT INTO symptoms (timestamp, symptom, severity, notes)
            VALUES (?, ?, ?, ?)
        ''', (
            symptom_entry['timestamp'],
            symptom_entry['symptom'],
            symptom_entry['severity'],
            symptom_entry['notes']
        ))
        self.conn.commit()
    
    def save_alert(self, alert_type, message, severity="medium"):
        """Enregistre une alerte dans la base de données"""
        timestamp = datetime.now()
        self.cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, alert_type, message, severity))
        self.conn.commit()
        
        # Ajouter à la liste des alertes en mémoire
        self.alerts.append({
            'timestamp': timestamp,
            'type': alert_type,
            'message': message,
            'severity': severity
        })
    
    def connect_sensors(self):
        """Simule la connexion à des capteurs médicaux réels"""
        # Dans une implémentation réelle, ceci se connecterait à des APIs de dispositifs médicaux
        print("Connexion aux capteurs médicaux...")
        time.sleep(2)
        print("Capteurs connectés et calibration terminée")
    
    def read_sensor_data(self):
        """Simule la lecture de données depuis des capteurs réels"""
        # Ces valeurs pourraient venir d'APIs comme:
        # - Oxymètres de pouls connectés
        # - Moniteurs de signes vitaux
        # - Dispositifs portables (smartwatches)
        
        spo2 = random.randint(88, 99)
        heart_rate = random.randint(60, 120)
        respiratory_rate = random.randint(12, 30)
        temperature = round(random.uniform(36.0, 38.5), 1)
        systolic_bp = random.randint(100, 160)
        diastolic_bp = random.randint(60, 100)
        activity_level = random.choice(['repos', 'léger', 'modéré', 'élevé'])
        
        return {
            'spo2': spo2,
            'heart_rate': heart_rate,
            'respiratory_rate': respiratory_rate,
            'temperature': temperature,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'activity_level': activity_level
        }
    
    def simulate_measurement(self):
        """Simule une lecture de capteurs avec des valeurs réalistes"""
        measurement = self.read_sensor_data()
        
        # Enregistrement avec horodatage
        current_time = datetime.now()
        measurement['time'] = current_time
        
        # Ajouter aux données en mémoire
        self.data['timestamp'].append(current_time)
        self.data['spo2'].append(measurement['spo2'])
        self.data['heart_rate'].append(measurement['heart_rate'])
        self.data['respiratory_rate'].append(measurement['respiratory_rate'])
        self.data['temperature'].append(measurement['temperature'])
        self.data['systolic_bp'].append(measurement['systolic_bp'])
        self.data['diastolic_bp'].append(measurement['diastolic_bp'])
        self.data['activity_level'].append(measurement['activity_level'])
        
        # Sauvegarder dans la base de données
        self.save_measurement(measurement)
        
        return measurement
    
    def log_symptom(self, symptom, severity, notes=""):
        """Enregistre un symptôme dans le journal"""
        entry = {
            'timestamp': datetime.now(),
            'symptom': symptom,
            'severity': severity,  # 1-10
            'notes': notes
        }
        self.symptom_log.append(entry)
        self.data['symptoms'].append(f"{symptom} (severity: {severity})")
        
        # Sauvegarder dans la base de données
        self.save_symptom(entry)
        
        # Vérification des exacerbations potentielles
        self.check_exacerbation(entry)
        
        return entry
    
    def check_exacerbation(self, symptom_entry):
        """Vérifie les signes d'exacerbation"""
        # Dernières mesures
        if len(self.data['spo2']) > 0:
            latest_spo2 = self.data['spo2'][-1]
            
            # Alertes basées sur les symptômes et la SpO2
            if (symptom_entry['symptom'].lower() in ['essoufflement', 'toux', 'fatigue'] and 
                symptom_entry['severity'] >= 7 and latest_spo2 < 92):
                self.send_alert("exacerbation", "Signes d'exacerbation possible")
                return True
                
        return False
    
    def send_alert(self, alert_type, message):
        """Envoie une alerte via multiple canaux"""
        # Enregistrer l'alerte
        severity = "high" if alert_type == "exacerbation" else "medium"
        self.save_alert(alert_type, message, severity)
        
        # Afficher dans la console
        print(f"🔴 ALERTE: {message} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Envoyer par email si configuré
        if self.config['email_alerts']['enabled']:
            self.send_email_alert(message, severity)
        
        # Dans une application réelle, on pourrait aussi envoyer un SMS ici
    
    def send_email_alert(self, message, severity):
        """Envoie une alerte par email"""
        try:
            sender_email = self.config['email_alerts']['sender_email']
            sender_password = self.config['email_alerts']['sender_password']
            recipient_emails = self.config['email_alerts']['recipient_emails']
            
            if not sender_email or not recipient_emails:
                return
            
            # Création du message
            email_message = MIMEMultipart()
            email_message['From'] = sender_email
            email_message['To'] = ", ".join(recipient_emails)
            email_message['Subject'] = f"Alerte Médicale - {self.patient_name} - {severity.upper()}"
            
            body = f"""
            Alerte pour le patient: {self.patient_name}
            Niveau de sévérité: {severity.upper()}
            Message: {message}
            Horodatage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Veuillez prendre les mesures appropriées.
            """
            email_message.attach(MIMEText(body, 'plain'))
            
            # Connexion au serveur SMTP et envoi
            with smtplib.SMTP(self.config['email_alerts']['smtp_server'], 
                             self.config['email_alerts']['smtp_port']) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_emails, email_message.as_string())
                
            print("📧 Email d'alerte envoyé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email: {str(e)}")
    
    def predict_deterioration(self, hours=6):
        """Prédit une détérioration potentielle basée sur les données historiques"""
        if len(self.data['spo2']) < self.config['prediction_settings']['window_size']:
            return "Données insuffisantes pour la prédiction"
        
        # Préparer les données pour la prédiction
        window_size = self.config['prediction_settings']['window_size']
        spo2_data = self.data['spo2'][-window_size:]
        
        # Créer un modèle de régression linéaire simple
        X = np.arange(len(spo2_data)).reshape(-1, 1)
        y = np.array(spo2_data)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédire les prochaines heures
        future_X = np.arange(len(spo2_data), len(spo2_data) + hours).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # Vérifier si une détérioration est prévue
        if any(pred < self.config['alert_thresholds']['spo2_low'] for pred in predictions):
            return f"Détérioration prévue dans les {hours} heures. SpO2 pourrait descendre sous {self.config['alert_thresholds']['spo2_low']}%"
        
        return "Aucune détérioration significative prévue"
    
    def detect_anomalies(self):
        """Détecte les anomalies dans les données vitales"""
        if len(self.data['spo2']) < 10:
            return "Données insuffisantes pour la détection d'anomalies"
        
        # Préparer les données
        vital_data = np.array([
            self.data['spo2'][-10:],
            self.data['heart_rate'][-10:],
            self.data['respiratory_rate'][-10:]
        ]).T
        
        # Entraîner un modèle de détection d'anomalies
        model = IsolationForest(contamination=0.1)
        model.fit(vital_data)
        
        # Prédire les anomalies
        predictions = model.predict(vital_data)
        
        # Compter les anomalies
        anomaly_count = sum(predictions == -1)
        
        if anomaly_count > 2:
            return f"{anomaly_count} anomalies détectées dans les signes vitaux"
        
        return "Aucune anomalie détectée"
    
    def generate_comprehensive_report(self, hours=24):
        """Génère un rapport complet avec prédictions et analyses"""
        # Récupérer les données de la base de données
        query = f"""
        SELECT * FROM measurements 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp
        """
        df = pd.read_sql_query(query, self.conn)
        
        if df.empty:
            return "Aucune donnée disponible pour cette période"
        
        # Générer des statistiques
        report = f"""
📊 RAPPORT COMPLET - {self.patient_name}
Période: {hours} heures
----------------------------------------
Valeurs moyennes:
- SpO2: {df['spo2'].mean():.1f}%
- Fréquence cardiaque: {df['heart_rate'].mean():.1f} bpm
- Fréquence respiratoire: {df['respiratory_rate'].mean():.1f} rpm
- Température: {df['temperature'].mean():.1f} °C
- Pression artérielle: {df['systolic_bp'].mean():.1f}/{df['diastolic_bp'].mean():.1f} mmHg

Valeurs extrêmes:
- SpO2 min: {df['spo2'].min()}%
- SpO2 max: {df['spo2'].max()}%
- FC max: {df['heart_rate'].max()} bpm
- FR max: {df['respiratory_rate'].max()} rpm

Épisodes de désaturation (SpO2 < {self.config['alert_thresholds']['spo2_low']}%): {len(df[df['spo2'] < self.config['alert_thresholds']['spo2_low']])}

Analyse de prédiction:
{self.predict_deterioration()}

Détection d'anomalies:
{self.detect_anomalies()}
        """
        
        return report
    
    def export_to_ehr(self, format='hl7'):
        """Exporte les données vers un format compatible avec les DME"""
        # Cette fonction simulerait l'exportation vers un système de dossier médical électronique
        # En pratique, cela utiliserait des standards comme HL7 FHIR
        
        print(f"Exportation des données au format {format.upper()} pour intégration DME...")
        # Simulation d'exportation
        time.sleep(2)
        print("Données exportées avec succès")
        
        return True

class MonitoringGUI:
    def __init__(self, monitor):
        self.monitor = monitor
        self.root = tk.Tk()
        self.root.title(f"Système de Surveillance - {monitor.patient_name}")
        self.setup_gui()
        
    def setup_gui(self):
        """Configure l'interface graphique"""
        # Cadre principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Informations patient
        ttk.Label(main_frame, text=f"Patient: {self.monitor.patient_name} (ID: {self.monitor.patient_id})", 
                 font=('Arial', 14, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Affichage des valeurs en temps réel
        self.spo2_var = tk.StringVar(value="SpO2: --%")
        self.hr_var = tk.StringVar(value="FC: -- bpm")
        self.rr_var = tk.StringVar(value="FR: -- rpm")
        self.temp_var = tk.StringVar(value="Temp: -- °C")
        self.bp_var = tk.StringVar(value="PA: --/-- mmHg")
        
        ttk.Label(main_frame, textvariable=self.spo2_var, font=('Arial', 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.hr_var, font=('Arial', 12)).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.rr_var, font=('Arial', 12)).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.temp_var, font=('Arial', 12)).grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(main_frame, textvariable=self.bp_var, font=('Arial', 12)).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Boutons
        ttk.Button(main_frame, text="Nouvelle Mesure", command=self.take_measurement).grid(row=4, column=0, pady=10)
        ttk.Button(main_frame, text="Rapport Complet", command=self.show_report).grid(row=4, column=1, pady=10)
        ttk.Button(main_frame, text="Journal Symptômes", command=self.show_symptom_log).grid(row=5, column=0, pady=10)
        ttk.Button(main_frame, text="Alertes", command=self.show_alerts).grid(row=5, column=1, pady=10)
        
        # Zone de texte pour les alertes
        self.alert_text = tk.Text(main_frame, height=10, width=50)
        self.alert_text.grid(row=6, column=0, columnspan=2, pady=10)
        self.alert_text.insert(tk.END, "Aucune alerte pour le moment\n")
        self.alert_text.config(state=tk.DISABLED)
        
        # Configuration de la mise à jour automatique
        self.update_interval = 30000  # 30 secondes
        self.update_display()
        
    def take_measurement(self):
        """Prend une nouvelle mesure et met à jour l'affichage"""
        measurement = self.monitor.simulate_measurement()
        self.update_display_values(measurement)
        
        # Vérifier les alertes
        self.check_measurement_alerts(measurement)
    
    def update_display_values(self, measurement):
        """Met à jour l'affichage avec les nouvelles valeurs"""
        self.spo2_var.set(f"SpO2: {measurement['spo2']}%")
        self.hr_var.set(f"FC: {measurement['heart_rate']} bpm")
        self.rr_var.set(f"FR: {measurement['respiratory_rate']} rpm")
        self.temp_var.set(f"Temp: {measurement['temperature']} °C")
        self.bp_var.set(f"PA: {measurement['systolic_bp']}/{measurement['diastolic_bp']} mmHg")
    
    def check_measurement_alerts(self, measurement):
        """Vérifie si la mesure déclenche des alertes"""
        thresholds = self.monitor.config['alert_thresholds']
        
        if measurement['spo2'] < thresholds['spo2_low']:
            self.monitor.send_alert("spo2_low", f"SpO2 basse: {measurement['spo2']}%")
        
        if measurement['heart_rate'] > thresholds['heart_rate_high']:
            self.monitor.send_alert("heart_rate_high", f"Tachycardie: {measurement['heart_rate']} bpm")
        
        if measurement['respiratory_rate'] > thresholds['respiratory_rate_high']:
            self.monitor.send_alert("respiratory_rate_high", f"Tachypnée: {measurement['respiratory_rate']} rpm")
        
        if measurement['temperature'] > thresholds['temperature_high']:
            self.monitor.send_alert("temperature_high", f"Fièvre: {measurement['temperature']} °C")
        
        self.update_alert_display()
    
    def update_alert_display(self):
        """Met à jour l'affichage des alertes"""
        self.alert_text.config(state=tk.NORMAL)
        self.alert_text.delete(1.0, tk.END)
        
        if not self.monitor.alerts:
            self.alert_text.insert(tk.END, "Aucune alerte pour le moment\n")
        else:
            for alert in self.monitor.alerts[-5:]:  # Afficher les 5 dernières alertes
                self.alert_text.insert(tk.END, 
                    f"{alert['timestamp'].strftime('%H:%M:%S')} - {alert['message']}\n")
        
        self.alert_text.config(state=tk.DISABLED)
    
    def show_report(self):
        """Affiche un rapport complet"""
        report = self.monitor.generate_comprehensive_report(24)
        report_window = tk.Toplevel(self.root)
        report_window.title("Rapport Complet")
        
        text_area = tk.Text(report_window, width=80, height=20)
        text_area.pack(padx=10, pady=10)
        text_area.insert(tk.END, report)
        text_area.config(state=tk.DISABLED)
    
    def show_symptom_log(self):
        """Affiche le journal des symptômes"""
        log_window = tk.Toplevel(self.root)
        log_window.title("Journal des Symptômes")
        
        # Récupérer les symptômes de la base de données
        query = "SELECT timestamp, symptom, severity, notes FROM symptoms ORDER BY timestamp DESC LIMIT 20"
        df = pd.read_sql_query(query, self.monitor.conn)
        
        if df.empty:
            text_area = tk.Text(log_window, width=80, height=10)
            text_area.pack(padx=10, pady=10)
            text_area.insert(tk.END, "Aucun symptôme enregistré")
            text_area.config(state=tk.DISABLED)
            return
        
        # Créer un tableau avec les symptômes
        tree = ttk.Treeview(log_window, columns=('Date', 'Symptôme', 'Sévérité', 'Notes'), show='headings')
        tree.heading('Date', text='Date')
        tree.heading('Symptôme', text='Symptôme')
        tree.heading('Sévérité', text='Sévérité')
        tree.heading('Notes', text='Notes')
        
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=(
                row['timestamp'],
                row['symptom'],
                row['severity'],
                row['notes']
            ))
        
        tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def show_alerts(self):
        """Affiche toutes les alertes"""
        alert_window = tk.Toplevel(self.root)
        alert_window.title("Historique des Alertes")
        
        # Récupérer les alertes de la base de données
        query = "SELECT timestamp, alert_type, message, severity FROM alerts ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, self.monitor.conn)
        
        if df.empty:
            text_area = tk.Text(alert_window, width=80, height=10)
            text_area.pack(padx=10, pady=10)
            text_area.insert(tk.END, "Aucune alerte enregistrée")
            text_area.config(state=tk.DISABLED)
            return
        
        # Créer un tableau avec les alertes
        tree = ttk.Treeview(alert_window, columns=('Date', 'Type', 'Message', 'Sévérité'), show='headings')
        tree.heading('Date', text='Date')
        tree.heading('Type', text='Type')
        tree.heading('Message', text='Message')
        tree.heading('Sévérité', text='Sévérité')
        
        for _, row in df.iterrows():
            tree.insert('', tk.END, values=(
                row['timestamp'],
                row['alert_type'],
                row['message'],
                row['severity']
            ))
        
        tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def update_display(self):
        """Met à jour périodiquement l'affichage"""
        self.take_measurement()
        self.root.after(self.update_interval, self.update_display)
    
    def run(self):
        """Lance l'interface graphique"""
        self.root.mainloop()

# Exemple d'utilisation
if __name__ == "__main__":
    # Création d'un moniteur pour un patient
    patient_monitor = AdvancedPatientMonitor("PAT123", "Jean Dupont")
    
    # Connexion aux capteurs (simulée)
    patient_monitor.connect_sensors()
    
    # Configuration des alertes email (à adapter avec de vraies valeurs)
    # patient_monitor.config['email_alerts']['enabled'] = True
    # patient_monitor.config['email_alerts']['sender_email'] = "votre_email@gmail.com"
    # patient_monitor.config['email_alerts']['sender_password'] = "votre_mot_de_passe"
    # patient_monitor.config['email_alerts']['recipient_emails'] = ["medecin@hopital.com"]
    
    # Lancement de l'interface graphique
    app = MonitoringGUI(patient_monitor)
    app.run()