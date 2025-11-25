"""
Enhanced HD Pharmacy Visualization
Professional-grade graphics with animations, shadows, and effects
"""

import pygame
import numpy as np
import math
from typing import Optional


class EnhancedPharmacyRenderer:
    """
    Professional HD visualization for Pharmacy Stock Environment
    Features: 3D-style graphics, animations, particle effects, shadows
    """
    
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        
        # HD display with anti-aliasing
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("PharmaStock AI - HD Professional Dashboard")
        
        self.clock = pygame.time.Clock()
        self.fps = 60  # Smooth 60 FPS
        
        # Professional Color Palette
        self.COLORS = {
            'background': (245, 247, 250),
            'primary': (41, 128, 185),
            'success': (46, 204, 113),
            'warning': (241, 196, 15),
            'danger': (231, 76, 60),
            'dark': (44, 62, 80),
            'light': (236, 240, 241),
            'white': (255, 255, 255),
            'shadow': (0, 0, 0, 30),
            'accent': (155, 89, 182),
            'info': (52, 152, 219),
            'med_blue': (100, 181, 246),
            'med_green': (129, 199, 132),
            'glass': (255, 255, 255, 180)
        }
        
        # Fonts - Professional hierarchy
        self.fonts = {
            'title': pygame.font.Font(None, 56),
            'heading': pygame.font.Font(None, 42),
            'subheading': pygame.font.Font(None, 32),
            'body': pygame.font.Font(None, 26),
            'small': pygame.font.Font(None, 20)
        }
        
        # Animation states
        self.frame = 0
        self.medicine_bottles = []
        self.patients = []
        self.particles = []
        self.alert_pulse = 0
        self.chart_history = []
        
        # Initialize animations
        self._init_medicine_bottles()
        self._init_patients()
        
    def _init_medicine_bottles(self):
        """Initialize animated medicine bottle sprites"""
        for i in range(5):
            bottle = {
                'x': 100 + i * 80,
                'y': 200,
                'height': 60,
                'color': self.COLORS['med_blue'],
                'bob_offset': i * 0.5,
                'visible': True
            }
            self.medicine_bottles.append(bottle)
    
    def _init_patients(self):
        """Initialize patient sprites for demand visualization"""
        for i in range(15):
            patient = {
                'x': 100 + (i % 5) * 80,
                'y': 450 + (i // 5) * 50,
                'active': False,
                'color': self.COLORS['primary'],
                'animation_offset': i * 0.3
            }
            self.patients.append(patient)
    
    def _draw_shadow(self, surface, rect, offset=5, alpha=30):
        """Draw soft shadow for depth effect"""
        shadow_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        shadow_rect = rect.copy()
        shadow_rect.x += offset
        shadow_rect.y += offset
        pygame.draw.rect(shadow_surf, (*self.COLORS['dark'][:3], alpha), 
                        (0, 0, rect.width, rect.height), border_radius=10)
        surface.blit(shadow_surf, shadow_rect)
    
    def _draw_glass_panel(self, x, y, width, height, title=""):
        """Draw modern glass-morphism panel"""
        # Shadow
        shadow_rect = pygame.Rect(x + 5, y + 5, width, height)
        shadow_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 20), (0, 0, width, height), border_radius=15)
        self.screen.blit(shadow_surf, shadow_rect)
        
        # Glass panel
        panel_surf = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, self.COLORS['glass'], (0, 0, width, height), border_radius=15)
        self.screen.blit(panel_surf, (x, y))
        
        # Border
        pygame.draw.rect(self.screen, self.COLORS['primary'], (x, y, width, height), 3, border_radius=15)
        
        # Title bar
        if title:
            title_height = 50
            title_surf = pygame.Surface((width, title_height), pygame.SRCALPHA)
            pygame.draw.rect(title_surf, (*self.COLORS['primary'][:3], 200), 
                           (0, 0, width, title_height), border_radius=15)
            self.screen.blit(title_surf, (x, y))
            
            title_text = self.fonts['subheading'].render(title, True, self.COLORS['white'])
            title_rect = title_text.get_rect(center=(x + width // 2, y + title_height // 2))
            self.screen.blit(title_text, title_rect)
    
    def _draw_header(self, day):
        """Draw professional header with gradient"""
        # Gradient background
        for i in range(150):
            alpha = 255 - int((i / 150) * 100)
            color = (*self.COLORS['primary'][:3], alpha)
            pygame.draw.rect(self.screen, color, (0, i, self.width, 1))
        
        # Title with shadow
        title_text = self.fonts['title'].render("PharmaStock AI", True, self.COLORS['white'])
        shadow_text = self.fonts['title'].render("PharmaStock AI", True, (0, 0, 0, 100))
        
        title_x = self.width // 2
        self.screen.blit(shadow_text, shadow_text.get_rect(center=(title_x + 3, 48)))
        self.screen.blit(title_text, title_text.get_rect(center=(title_x, 45)))
        
        # Subtitle
        subtitle = self.fonts['body'].render("Intelligent Inventory Management System", True, self.COLORS['light'])
        self.screen.blit(subtitle, subtitle.get_rect(center=(title_x, 90)))
        
        # Day counter with icon
        day_panel_x = self.width - 250
        day_panel_y = 30
        pygame.draw.rect(self.screen, self.COLORS['accent'], 
                        (day_panel_x, day_panel_y, 200, 60), border_radius=30)
        
        day_text = self.fonts['heading'].render(f"Day {day}/30", True, self.COLORS['white'])
        self.screen.blit(day_text, day_text.get_rect(center=(day_panel_x + 100, day_panel_y + 30)))
    
    def _draw_stock_visualization(self, stock, stockouts, max_stock=50):
        """Draw enhanced stock visualization with 3D bottles"""
        panel_x, panel_y = 50, 160
        panel_width, panel_height = 550, 280
        
        self._draw_glass_panel(panel_x, panel_y, panel_width, panel_height, "Medicine Inventory")
        
        # Content area
        content_y = panel_y + 70
        
        # Animated medicine bottles
        bottles_to_show = min(int(stock / 10) + 1, 5)
        for i, bottle in enumerate(self.medicine_bottles):
            if i < bottles_to_show:
                # Bobbing animation
                bob = math.sin(self.frame * 0.1 + bottle['bob_offset']) * 5
                bottle_x = panel_x + 80 + i * 90
                bottle_y = content_y + 50 + bob
                
                # 3D bottle effect
                # Shadow
                pygame.draw.ellipse(self.screen, (0, 0, 0, 50),
                                  (bottle_x, bottle_y + 70, 50, 15))
                
                # Bottle body
                pygame.draw.rect(self.screen, self.COLORS['med_blue'],
                               (bottle_x, bottle_y, 50, 60), border_radius=5)
                pygame.draw.rect(self.screen, self.COLORS['primary'],
                               (bottle_x, bottle_y, 50, 60), 2, border_radius=5)
                
                # Cap
                pygame.draw.rect(self.screen, self.COLORS['danger'],
                               (bottle_x + 15, bottle_y - 10, 20, 10), border_radius=3)
                
                # Shine effect
                shine_surf = pygame.Surface((15, 30), pygame.SRCALPHA)
                pygame.draw.rect(shine_surf, (255, 255, 255, 100), (0, 0, 15, 30), border_radius=3)
                self.screen.blit(shine_surf, (bottle_x + 5, bottle_y + 10))
        
        # Stock level bar
        bar_x = panel_x + 30
        bar_y = content_y + 150
        bar_width = panel_width - 60
        bar_height = 50
        
        # Background
        pygame.draw.rect(self.screen, self.COLORS['light'],
                        (bar_x, bar_y, bar_width, bar_height), border_radius=25)
        
        # Fill with gradient
        fill_width = int((stock / max_stock) * bar_width)
        if stock < 15:
            bar_color = self.COLORS['danger']
        elif stock < 30:
            bar_color = self.COLORS['warning']
        else:
            bar_color = self.COLORS['success']
        
        for i in range(fill_width):
            color_factor = 0.8 + (i / fill_width) * 0.2
            color = tuple(int(c * color_factor) for c in bar_color[:3])
            pygame.draw.rect(self.screen, color, (bar_x + i, bar_y, 1, bar_height))
        
        # Border
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, bar_width, bar_height), 4, border_radius=25)
        
        # Stock number
        stock_text = self.fonts['heading'].render(f"{stock} / {max_stock}", True, self.COLORS['dark'])
        stock_rect = stock_text.get_rect(center=(panel_x + panel_width // 2, bar_y + bar_height // 2))
        self.screen.blit(stock_text, stock_rect)
        
        # Units label
        units_text = self.fonts['small'].render("units in stock", True, self.COLORS['dark'])
        self.screen.blit(units_text, units_text.get_rect(center=(panel_x + panel_width // 2, bar_y + bar_height + 25)))
    
    def _draw_demand_visualization(self, demand):
        """Draw animated patient demand"""
        panel_x, panel_y = 650, 160
        panel_width, panel_height = 580, 280
        
        self._draw_glass_panel(panel_x, panel_y, panel_width, panel_height, "Today's Demand")
        
        content_y = panel_y + 70
        
        # Activate patients based on demand
        for i, patient in enumerate(self.patients):
            patient['active'] = i < demand
        
        # Draw patients with animation
        for i, patient in enumerate(self.patients):
            if patient['active']:
                # Walking animation
                walk_offset = math.sin(self.frame * 0.15 + patient['animation_offset']) * 3
                patient_x = panel_x + 50 + (i % 8) * 65
                patient_y = content_y + 20 + (i // 8) * 70 + walk_offset
                
                # Shadow
                pygame.draw.ellipse(self.screen, (0, 0, 0, 40),
                                  (patient_x + 5, patient_y + 50, 30, 10))
                
                # Patient body
                # Head
                pygame.draw.circle(self.screen, self.COLORS['info'], 
                                 (int(patient_x + 15), int(patient_y)), 12)
                pygame.draw.circle(self.screen, self.COLORS['dark'],
                                 (int(patient_x + 15), int(patient_y)), 12, 2)
                
                # Body
                pygame.draw.rect(self.screen, self.COLORS['primary'],
                               (patient_x + 5, patient_y + 10, 20, 25), border_radius=5)
                
                # Arms
                pygame.draw.circle(self.screen, self.COLORS['primary'],
                                 (int(patient_x), int(patient_y + 20)), 6)
                pygame.draw.circle(self.screen, self.COLORS['primary'],
                                 (int(patient_x + 30), int(patient_y + 20)), 6)
                
                # Legs
                pygame.draw.rect(self.screen, self.COLORS['dark'],
                               (patient_x + 8, patient_y + 35, 6, 15), border_radius=3)
                pygame.draw.rect(self.screen, self.COLORS['dark'],
                               (patient_x + 16, patient_y + 35, 6, 15), border_radius=3)
        
        # Demand counter
        demand_text = self.fonts['heading'].render(f"{demand} Patients", True, self.COLORS['primary'])
        self.screen.blit(demand_text, demand_text.get_rect(center=(panel_x + panel_width // 2, content_y + 200)))
    
    def _draw_metrics_dashboard(self, reward, service_level, stockouts, action, pending_order):
        """Draw comprehensive metrics dashboard"""
        panel_x, panel_y = 50, 470
        panel_width, panel_height = 1180, 200
        
        self._draw_glass_panel(panel_x, panel_y, panel_width, panel_height, "Performance Metrics")
        
        content_y = panel_y + 70
        
        # Metric cards
        metrics = [
            {
                'label': 'Reward',
                'value': f"{reward:.1f}",
                'color': self.COLORS['success'] if reward >= 0 else self.COLORS['danger'],
                'icon': '💰'
            },
            {
                'label': 'Service Level',
                'value': f"{service_level*100:.0f}%",
                'color': self.COLORS['success'] if service_level >= 0.8 else self.COLORS['warning'],
                'icon': '📊'
            },
            {
                'label': 'Stockouts',
                'value': f"{stockouts}",
                'color': self.COLORS['success'] if stockouts == 0 else self.COLORS['danger'],
                'icon': '⚠️'
            },
            {
                'label': 'Action',
                'value': f"Order {[0, 5, 10, 20][min(action, 3)]}",
                'color': self.COLORS['primary'],
                'icon': '📦'
            },
            {
                'label': 'Pending',
                'value': f"{pending_order}",
                'color': self.COLORS['accent'],
                'icon': '🚚'
            }
        ]
        
        card_width = 210
        spacing = 20
        start_x = panel_x + 40
        
        for i, metric in enumerate(metrics):
            card_x = start_x + i * (card_width + spacing)
            card_y = content_y
            
            # Card background
            card_surf = pygame.Surface((card_width, 100), pygame.SRCALPHA)
            pygame.draw.rect(card_surf, (*metric['color'][:3], 30), 
                           (0, 0, card_width, 100), border_radius=10)
            self.screen.blit(card_surf, (card_x, card_y))
            pygame.draw.rect(self.screen, metric['color'],
                           (card_x, card_y, card_width, 100), 3, border_radius=10)
            
            # Icon
            icon_text = self.fonts['heading'].render(metric['icon'], True, metric['color'])
            self.screen.blit(icon_text, (card_x + 10, card_y + 10))
            
            # Value
            value_text = self.fonts['heading'].render(metric['value'], True, metric['color'])
            value_rect = value_text.get_rect(center=(card_x + card_width // 2, card_y + 45))
            self.screen.blit(value_text, value_rect)
            
            # Label
            label_text = self.fonts['small'].render(metric['label'], True, self.COLORS['dark'])
            label_rect = label_text.get_rect(center=(card_x + card_width // 2, card_y + 80))
            self.screen.blit(label_text, label_rect)
    
    def _draw_alert_animation(self):
        """Draw pulsing stockout alert - SMALL AND CENTERED (NO FULL SCREEN OVERLAY)"""
        self.alert_pulse = (self.alert_pulse + 0.1) % (2 * math.pi)
        alpha = int(128 + 127 * math.sin(self.alert_pulse))
        
        # Dimensions for a small, rounded alert box
        alert_width = 320
        alert_height = 60
        alert_x = (self.width - alert_width) // 2  # Center horizontally
        alert_y = 105  # Position it just under the title/header
        
        # Create a surface for the alert box with alpha transparency
        alert_surf = pygame.Surface((alert_width, alert_height), pygame.SRCALPHA)
        
        # Draw the rounded red box
        pygame.draw.rect(alert_surf, (*self.COLORS['danger'][:3], alpha), 
                        (0, 0, alert_width, alert_height), border_radius=30)
        
        # Add a white border to make it pop
        pygame.draw.rect(alert_surf, (255, 255, 255, 200), 
                        (0, 0, alert_width, alert_height), 3, border_radius=30)
        
        # Blit the box onto the screen
        self.screen.blit(alert_surf, (alert_x, alert_y))
        
        # Draw the alert text centered in the box
        alert_text = self.fonts['subheading'].render("⚠️ LOW STOCK ALERT ⚠️", True, self.COLORS['white'])
        text_rect = alert_text.get_rect(center=(alert_x + alert_width // 2, alert_y + alert_height // 2))
        self.screen.blit(alert_text, text_rect)
    
    def render(self, env_state):
        """Main render function"""
        self.frame += 1
        
        # Background
        self.screen.fill(self.COLORS['background'])
        
        # Extract state
        stock = env_state.get('stock', 0)
        demand = env_state.get('demand', 0)
        day = env_state.get('day', 0)
        action = env_state.get('action', 0)
        reward = env_state.get('reward', 0)
        stockouts = env_state.get('stockouts', 0)
        service_level = env_state.get('service_level', 1.0)
        pending_order = env_state.get('pending_order', 0)
        
        # Draw components
        self._draw_header(day)
        self._draw_stock_visualization(stock, stockouts)
        self._draw_demand_visualization(demand)
        self._draw_metrics_dashboard(reward, service_level, stockouts, action, pending_order)
        
        # Alert animation - Modified to be subtle
        if stockouts > 0 or stock < 5:
            self._draw_alert_animation()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
    
    def close(self):
        """Clean up"""
        pygame.quit()


# Test the enhanced renderer
if __name__ == "__main__":
    print("Testing Enhanced HD Renderer...")
    
    renderer = EnhancedPharmacyRenderer()
    
    # Simulate environment states
    running = True
    frame = 0
    
    while running and frame < 300:
        stock = int(25 + 15 * math.sin(frame * 0.1))
        demand = int(8 + 5 * math.sin(frame * 0.05))
        
        env_state = {
            'stock': stock,
            'demand': demand,
            'day': frame // 10,
            'action': (frame // 30) % 4,
            'reward': np.random.uniform(-10, 10),
            'stockouts': 1 if stock < 10 else 0,
            'service_level': stock / 50,
            'pending_order': 5 if (frame % 50) < 25 else 0
        }
        
        running = renderer.render(env_state)
        frame += 1
    
    renderer.close()
    print("✅ Enhanced renderer test complete!")