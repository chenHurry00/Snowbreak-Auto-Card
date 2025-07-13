import cv2
import numpy as np
import pyautogui
import time
import os
import random
from typing import List, Tuple, Dict, Optional
from enum import Enum
import threading
import json

class CardType(Enum):
    """卡牌类型枚举"""
    LIGHTNING = "lightning"  # 闪电
    SNOWFLAKE = "snowflake"  # 雪花
    FIRE = "fire"           # 火焰
    KETTLE = "kettle"       # 热水壶

class PlayerState(Enum):
    """玩家状态枚举"""
    NORMAL1 = "normal1"       # 正常
    DRUNK1 = "drunk1"         # 醉酒倒下
    NORMAL2 = "normal2"       
    DRUNK2 = "drunk2"         
    NORMAL3 = "normal3"       
    DRUNK3 = "drunk3"         

class GameAutomation:
    """游戏自动化主类"""
    
    def __init__(self):
        # 初始化配置
        self.config = {
            'screen_width': 1920,
            'screen_height': 1080,
            'card_templates_dir': 'card_templates',
            'player_templates_dir': 'player_templates',
            'confidence_threshold': 0.7,
            'detection_interval': 2,  # 检测间隔（秒）
            'card_region_threshold': 0.4,  # 卡牌区域重叠阈值
            # 手牌检测配置
            'card_width': 190,  # 固定卡牌宽度
            'card_height': 281,  # 固定卡牌高度
            'sliding_window_step': 150,  # 滑动窗口步长
            'card_overlap_threshold': 0.3,  # 卡牌重叠阈值
            'template_ratio': 1.35, # 模板放大倍率，用于匹配
            'rotation_angles': [-12, -7, 0, 5, 12],  # 旋转角度配置,滑动会取5张图
            # Debug图片输出
            'debug_export': 0

        }
        
        # 游戏区域配置
        self.game_areas = {
            'target_card_area': (27, 885, 105, 990),  # 左下角目标卡牌区域 (x1, y1, x2, y2)
            'hand_cards_area': (560, 625, 1350, 906),  # 屏幕中央手牌区域
            'one_more_area': (1579,949,1891,1041), # 再来一局区域
            'player_positions': {
                'player1': (0, 240, 430, 870),   # 玩家1位置
                'player2': (810, 200, 1150, 610), # 玩家2位置
                'player3': (1495, 190, 1925, 790),   # 玩家3位置
            },
            'confirm_position':(1760,955),
            'one_more_position': (1730,990)
        }
        
        # 存储模板图像
        self.card_templates = {}
        self.player_templates = {}
        self.game_status = {}
        
        # 当前游戏状态
        self.current_state = {
            'target_card': None,
            'hand_cards': [],
            'player_states': {},
            'is_my_turn': False,
            'is_one_more': False
        }
        
        # 初始化PyAutoGUI
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # 加载模板图像
        self.load_templates()
        
    def load_templates(self):
        """加载卡牌和玩家状态模板图像"""
        try:
            # 加载卡牌模板
            for card_type in CardType:
                template_path = os.path.join(self.config['card_templates_dir'], f'{card_type.value}.png')
                if os.path.exists(template_path):
                    self.card_templates[card_type] = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    print(f"已加载卡牌模板: {card_type.value}")
                else:
                    print(f"警告: 找不到卡牌模板 {template_path}")
            
            # 加载玩家状态模板
            for state in PlayerState:
                template_path = os.path.join(self.config['player_templates_dir'], f'{state.value}.png')
                if os.path.exists(template_path):
                    self.player_templates[state] = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    print(f"已加载玩家状态模板: {state.value}")
                else:
                    print(f"警告: 找不到玩家状态模板 {template_path}")

            # 加载游戏状态模板
            template_path = os.path.join(self.config['player_templates_dir'], 'one_more.png')
            if os.path.exists(template_path):
                self.game_status['one_more'] = cv2.imread(template_path, cv2.IMREAD_COLOR)
                print(f"已加载游戏状态模板: {state.value}")
            else:
                print(f"警告: 找不到游戏状态模板 {template_path}")
                    
        except Exception as e:
            print(f"加载模板时出错: {e}")
    
    def capture_screen(self) -> np.ndarray:
        """截取屏幕图像"""
        try:
            screenshot = pyautogui.screenshot()
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"截屏失败: {e}")
            return None
    
    def template_match(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = None) -> Tuple[bool, Tuple[int, int], float]:
        """模板匹配函数"""
        if threshold is None:
            threshold = self.config['confidence_threshold']
            
        # 执行模板匹配
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 判断是否匹配成功
        if max_val >= threshold:
            return True, max_loc, max_val
        else:
            return False, None, max_val
    
    def detect_target_card(self, screen: np.ndarray) -> Optional[CardType]:
        """检测左下角目标卡牌（无匹配时返回置信度最高项）"""
        x1, y1, x2, y2 = self.game_areas['target_card_area']
        target_area = screen[y1:y2, x1:x2]
        best_match = None
        best_confidence = 0
        all_matches = []  # 存储所有匹配结果[2,4]
        
        for card_type, template in self.card_templates.items():
            if template is None:
                continue
                
            matched, position, confidence = self.template_match(target_area, template)
            all_matches.append((card_type, confidence))  # 记录所有结果
            
            if confidence > best_confidence:  # 始终更新最佳置信度
                best_match = card_type
                best_confidence = confidence
        
        # 无完美匹配时选择置信度最高项（即使低于阈值）
        if best_match:
            print(f"检测到目标卡牌: {best_match.value} (置信度: {best_confidence:.2f})")
        else:
            # 从所有结果中筛选最佳（可能包含低置信度匹配）
            best_match = max(all_matches, key=lambda x: x[1])[0] if all_matches else None
            print(f"无精确匹配，采用最高置信度卡牌: {best_match.value} ({best_confidence:.2f})")
        
        return best_match
    
    def detect_player_states(self, screen: np.ndarray) -> Dict[str, PlayerState]:
        """检测三个玩家的状态"""
        player_states = {}
        
        # 定义每个玩家对应的状态模板
        player_templates_mapping = {
            'player1': {'normal': PlayerState.NORMAL1, 'drunk': PlayerState.DRUNK1},
            'player2': {'normal': PlayerState.NORMAL2, 'drunk': PlayerState.DRUNK2},
            'player3': {'normal': PlayerState.NORMAL3, 'drunk': PlayerState.DRUNK3}
        }
        
        for player_name, (x1, y1, x2, y2) in self.game_areas['player_positions'].items():
            player_area = screen[y1:y2, x1:x2]

            # 获取该玩家对应的状态模板
            player_template_states = player_templates_mapping[player_name]

            # 保存图片以供测试
            #cv2.imwrite(f"{player_name}_area_.png", player_area)
            #print(f"已保存玩家区域图像: {player_name}_area_.png")
            
            best_match = None
            best_confidence = 0
            
            # 检测该玩家的醉酒状态
            drunk_state = player_template_states['drunk']
            if drunk_state in self.player_templates:
                drunk_template = self.player_templates[drunk_state]
                matched, position, confidence = self.template_match(player_area, drunk_template)
                #print("醉酒匹配置信度: ",matched,confidence)
                if matched and confidence > best_confidence:
                    best_match = drunk_state
                    best_confidence = confidence
            
            # 检测该玩家的正常状态
            normal_state = player_template_states['normal']
            if normal_state in self.player_templates:
                normal_template = self.player_templates[normal_state]
                matched, position, confidence = self.template_match(player_area, normal_template)
                #print("正常匹配置信度: ",matched,confidence)
                
                if matched and confidence > best_confidence:
                    best_match = normal_state
                    best_confidence = confidence
            
            # 如果没有匹配到任何状态，默认为正常状态
            if best_match is None:
                best_match = player_template_states['normal']
                print(f"{player_name} 状态未识别，默认为正常状态")
            else:
                status = "醉酒倒下" if "drunk" in best_match.value else "正常"
                print(f"{player_name} 状态: {status} (置信度: {best_confidence:.2f})")
            
            player_states[player_name] = best_match
        
        return player_states
    
    def save_debug_image(self, image: np.ndarray, filename: str, annotation: str = ""):
        """保存调试图像"""
        if self.config['debug_export'] == 0:
            return 

        debug_path = os.path.join('debug_images', filename)
        
        # 注释添加到图像上
        if annotation:
            debug_image = image.copy()
            cv2.putText(debug_image, annotation, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(debug_path, debug_image)
        else:
            cv2.imwrite(debug_path, image)
        
        print(f"已保存调试图像: {debug_path}")

    def rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像函数（带边界处理）"""
        h, w = img.shape[:2]
        center = (w//2, h//2)
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算新边界尺寸
        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        
        # 调整旋转矩阵中心点
        M[0, 2] += (new_w - w) // 2
        M[1, 2] += (new_h - h) // 2
        
        # 执行旋转（黑色背景填充）
        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # 裁剪回原始尺寸（居中裁剪）
        crop_x = (new_w - w) // 2
        crop_y = (new_h - h) // 2
        return rotated[crop_y:crop_y+h, crop_x:crop_x+w]

    def detect_hand_cards(self, screen: np.ndarray) -> List[Dict]:
        """检测手牌 - 使用固定尺寸滑动窗口"""
        x1, y1, x2, y2 = self.game_areas['hand_cards_area']
        hand_area = screen[y1:y2, x1:x2]
        
        # 保存手牌区域原始图像
        self.save_debug_image(hand_area, "hand_area_original.png", "Hand Cards Area")
        
        detected_cards = []
        debug_image = hand_area.copy()
        
        # 计算滑动窗口的范围
        area_height, area_width = hand_area.shape[:2]
        card_width = self.config['card_width']
        card_height = self.config['card_height']
        step = self.config['sliding_window_step']
        rotation_angles = self.config['rotation_angles']
        rotation_angles_index = 0
        
        print(f"手牌区域尺寸: {area_width}x{area_height}")
        print(f"卡牌尺寸: {card_width}x{card_height}")
        print(f"滑动步长: {step}")
        
        # 滑动窗口检测
        for x in range(0, area_width - card_width + 1, step):
            for y in range(0, area_height - card_height + 1, step):
                # 提取当前窗口
                window = hand_area[y:y+card_height, x:x+card_width]
                rotated_window = self.rotate_image(window, rotation_angles[rotation_angles_index])
                
                # 检测每种卡牌类型
                best_card_type = None
                best_confidence = 0
                
                for card_type, template in self.card_templates.items():
                    if template is None:
                        continue

                    # 将窗口缩放到模板尺寸
                    A = int(template.shape[1]*self.config['template_ratio'])
                    B = int(template.shape[0]*self.config['template_ratio'])
                    resized_window = cv2.resize(rotated_window, (A, B))
                    self.save_debug_image(resized_window, f"hand_area_window_{y+card_height}{x+card_width}.png")
                    
                    matched, position, confidence = self.template_match(resized_window, template)
                    #print(f"{card_type}置信度：{confidence}")
                    
                    if confidence > best_confidence:
                        best_card_type = card_type
                        best_confidence = confidence
                
                # 如果找到匹配的卡牌
                if best_card_type and best_confidence >= self.config['confidence_threshold']:
                    card_info = {
                        'type': best_card_type,
                        'position': (x + x1, y + y1),  # 转换为屏幕坐标
                        'window_position': (x, y),     # 在手牌区域中的位置
                        'size': (card_width, card_height),
                        'confidence': best_confidence,
                        'center': (x + card_width//2, y + card_height//2)
                    }
                    
                    # 检查是否与已检测的卡牌重叠
                    if not self.is_overlapping_with_existing(card_info, detected_cards):
                        detected_cards.append(card_info)
                        
                        # 在调试图像上绘制检测框
                        cv2.rectangle(debug_image, (x, y), (x + card_width, y + card_height), 
                                    (0, 255, 0), 2)
                        cv2.putText(debug_image, f"{best_card_type.value[:4]}", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(debug_image, f"{best_confidence:.2f}", 
                                  (x, y + card_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # print(f"检测到卡牌: {best_card_type.value} 位置({x}, {y}) 置信度: {best_confidence:.3f}")
                        
            rotation_angles_index += 1    
                
        
        # 保存带检测框的调试图像
        self.save_debug_image(debug_image, "hand_cards_detection.png", f"Detected {len(detected_cards)} cards")
        
        # 按x坐标排序（从左到右）
        detected_cards.sort(key=lambda card: card['window_position'][0])
        
        # 重新分配索引
        for i, card in enumerate(detected_cards):
            card['index'] = i
        
        # 保存单独的卡牌图像用于调试
        for i, card in enumerate(detected_cards):
            x, y = card['window_position']
            card_image = hand_area[y:y+card_height, x:x+card_width]
            self.save_debug_image(card_image, f"detected_card_{i}_{card['type'].value}.png", 
                                f"Card {i}: {card['type'].value} (conf: {card['confidence']:.3f})")
        
        print(f"\n总共检测到 {len(detected_cards)} 张手牌")
        for card in detected_cards:
            print(f"  卡牌 {card['index']}: {card['type'].value} "
                  f"(置信度: {card['confidence']:.3f}, 位置: {card['window_position']})")
        
        return detected_cards
    
    def is_overlapping_with_existing(self, new_card: Dict, existing_cards: List[Dict]) -> bool:
        """检查新卡牌是否与现有卡牌重叠"""
        new_x, new_y = new_card['window_position']
        card_width = self.config['card_width']
        card_height = self.config['card_height']
        
        for existing_card in existing_cards:
            exist_x, exist_y = existing_card['window_position']
            
            # 计算重叠区域
            overlap_x = max(0, min(new_x + card_width, exist_x + card_width) - max(new_x, exist_x))
            overlap_y = max(0, min(new_y + card_height, exist_y + card_height) - max(new_y, exist_y))
            overlap_area = overlap_x * overlap_y
            
            # 计算重叠比例
            card_area = card_width * card_height
            overlap_ratio = overlap_area / card_area
            
            if overlap_ratio > self.config['card_overlap_threshold']:
                # 如果重叠且新卡牌置信度更低，则认为重叠
                if new_card['confidence'] <= existing_card['confidence']:
                    return True
        
        return False
    
    def is_one_more(self,screen: np.ndarray) -> bool:
        x1, y1, x2, y2 = self.game_areas['one_more_area']
        threshold = self.config['confidence_threshold']

        area = screen[y1:y2, x1:x2]
        template = self.game_status['one_more']
        resized_window = cv2.resize(area, (template.shape[1], template.shape[0]))
        matched, position, confidence = self.template_match(resized_window, template)

        result = confidence > threshold
        if result:
            print("检测到再来一局")
        return result
    
    def is_my_turn(self, screen: np.ndarray) -> bool:
        """判断是否轮到玩家回合"""
        # 检测屏幕中央是否有卡牌出现
        hand_cards = self.detect_hand_cards(screen)
        return len(hand_cards) > 0
    
    def decide_action(self, target_card: CardType=[], hand_cards: List[Dict]=[], 
                     player_states: Dict[str, PlayerState]=[]) -> Dict:
        """决策要执行的动作"""
        # 这里实现你的游戏逻辑
        # 根据目标卡牌和手牌决定要打出什么牌
        # 目前的逻辑很简单，只是优先出对应牌
        
        action = {
            'type': 'play_card',
            'card_index': None,
            'target_player': None
        }

        if self.current_state['is_one_more']:
            action['type'] = 'one_more'
            return action
        
        # 决策逻辑示例
        if target_card and hand_cards:
            # 寻找匹配的卡牌
            for card in hand_cards:
                if card['type'] == target_card or card['type'].value == 'kettle':
                    action['card_index'] = card['index']
                    action['card_position'] = card['position']
                    action['card_size'] = card['size']
                    break
            
            # 如果没有匹配的卡牌，选择第一张
            if action['card_index'] is None and hand_cards:
                first_card = hand_cards[0]
                action['card_index'] = first_card['index']
                action['card_position'] = first_card['position']
                action['card_size'] = first_card['size']
        
        return action

    '''有判断为机器人的风险，改为使用贝塞尔曲线平滑执行
    def execute_action(self, action: Dict):
        """执行游戏动作"""
        if action['type'] == 'play_card' and action['card_index'] is not None:
            # 点击选中的卡牌
            card_pos = action['card_position']
            card_size = action['card_size']
            
            # 点击卡牌中心
            click_x = card_pos[0] + card_size[0] // 2
            click_y = card_pos[1] + card_size[1] // 2
            
            print(f"点击卡牌 {action['card_index']} 位置: ({click_x}, {click_y})")
            
            # 执行点击
            pyautogui.click(click_x, click_y)
            time.sleep(0.5)
            
            # 点击确认按钮
            confirm_x,confirm_y = self.game_areas['confirm_position']
            
            print(f"点击确认按钮位置: ({confirm_x}, {confirm_y})")
            pyautogui.click(confirm_x, confirm_y)
    '''
    def bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int], control_points: int = 2) -> list:
            """生成贝塞尔曲线路径点"""
            # 随机生成控制点，增加轨迹的随机性
            control_x1 = start[0] + random.randint(-100, 100)
            control_y1 = start[1] + random.randint(-50, 50)
            control_x2 = end[0] + random.randint(-100, 100)
            control_y2 = end[1] + random.randint(-50, 50)
            
            # 贝塞尔曲线的控制点
            points = [start, (control_x1, control_y1), (control_x2, control_y2), end]
            
            # 生成曲线上的点
            curve_points = []
            steps = random.randint(15, 25)  # 随机步数
            
            for i in range(steps + 1):
                t = i / steps
                # 三次贝塞尔曲线公式
                x = int((1-t)**3 * points[0][0] + 3*(1-t)**2*t * points[1][0] + 
                        3*(1-t)*t**2 * points[2][0] + t**3 * points[3][0])
                y = int((1-t)**3 * points[0][1] + 3*(1-t)**2*t * points[1][1] + 
                        3*(1-t)*t**2 * points[2][1] + t**3 * points[3][1])
                curve_points.append((x, y))
            
            return curve_points

    def human_like_click(self, target_x: int, target_y: int, duration: float = None):
        """模拟人类鼠标点击行为"""
        # 获取当前鼠标位置
        current_x, current_y = pyautogui.position()
        
        # 如果距离很近，直接点击
        distance = ((target_x - current_x)**2 + (target_y - current_y)**2)**0.5
        if distance < 5:
            pyautogui.click(target_x, target_y)
            return
        
        # 随机移动时间
        if duration is None:
            duration = random.uniform(0.05, 0.15)
        
        # 生成贝塞尔曲线路径
        path_points = self.bezier_curve((current_x, current_y), (target_x, target_y))
        
        # 沿路径移动鼠标
        for i, (x, y) in enumerate(path_points):
            # 添加轻微的随机偏移
            x += random.randint(-2, 2)
            y += random.randint(-2, 2)
            
            # 移动鼠标
            pyautogui.moveTo(x, y)
            
            # 变化的移动速度，开始慢，中间快，结束慢
            progress = i / len(path_points)
            if progress < 0.3:
                delay = duration * 0.015
            elif progress < 0.7:
                delay = duration * 0.005
            else:
                delay = duration * 0.015
                
            time.sleep(delay)
        
        # 到达目标后稍作停顿
        time.sleep(random.uniform(0.05, 0.15))
        
        # 执行点击
        pyautogui.click(target_x, target_y)
        
        # 点击后轻微移动，模拟人类行为
        offset_x = random.randint(-3, 3)
        offset_y = random.randint(-3, 3)
        pyautogui.moveRel(offset_x, offset_y)

    def execute_action(self, action: dict):
        """执行游戏动作 - 使用人性化鼠标轨迹"""
        if action['type'] == 'play_card' and action['card_index'] is not None:
            # 点击选中的卡牌
            card_pos = action['card_position']
            card_size = action['card_size']
            
            # 点击卡牌中心（加入随机偏移）
            click_x = card_pos[0] + card_size[0] // 2 + random.randint(-10, 10)
            click_y = card_pos[1] + card_size[1] // 2 + random.randint(-5, 5)
            
            print(f"点击卡牌 {action['card_index']} 位置: ({click_x}, {click_y})")
            
            # 使用人性化点击
            self.human_like_click(click_x, click_y, duration=random.uniform(0.4, 0.7))
            
            # 随机等待时间
            time.sleep(random.uniform(0.15, 0.3))
            
            # 点击确认按钮
            confirm_x, confirm_y = self.game_areas['confirm_position']
            confirm_x += random.randint(-5, 5)  # 添加随机偏移
            confirm_y += random.randint(-5, 5)
            
            print(f"点击确认按钮位置: ({confirm_x}, {confirm_y})")
            self.human_like_click(confirm_x, confirm_y, duration=random.uniform(0.3, 0.6))
            
            # 操作完成后稍作停顿
            time.sleep(random.uniform(0.2, 0.5))

        elif action['type'] == 'one_more':
            # 点击再来一局按钮
            one_more_x, one_more_y = self.game_areas['one_more_position']
            one_more_x += random.randint(-5, 5)  # 添加随机偏移
            one_more_y += random.randint(-5, 5)
            
            print(f"点击确认按钮位置: ({one_more_x}, {one_more_y})")
            self.human_like_click(one_more_x, one_more_y, duration=random.uniform(0.3, 0.6))
            
            # 操作完成后稍作停顿
            time.sleep(random.uniform(0.2, 0.5))

    def run_detection_loop(self,circle_times = 10):
        """主要的检测循环"""
        print("开始游戏检测循环...")
        times = 0
        while times<=circle_times:
            try:
                #默认喝酒：每轮开始前依次按下Q W E
                for key in ['q', 'w', 'e']:
                    pyautogui.keyDown(key)
                    time.sleep(0.1)  # 短暂按下保持
                    pyautogui.keyUp(key)
                    time.sleep(0.2)  # 按键间隔

                # 截取屏幕
                screen = self.capture_screen()
                if screen is None:
                    time.sleep(self.config['detection_interval'])
                    continue

                # 检测是否新开一局
                is_one_more = False
                is_my_turn = False
                is_one_more = self.is_one_more(screen)
                is_my_turn = self.is_my_turn(screen)
                self.current_state['is_one_more'] = is_one_more
                
                if is_my_turn:
                    # 检测目标卡牌
                    target_card = self.detect_target_card(screen)
                    self.current_state['target_card'] = target_card
                    
                    # 检测玩家状态
                    player_states = self.detect_player_states(screen)
                    self.current_state['player_states'] = player_states
                    self.current_state['is_my_turn'] = is_my_turn

                    # 检测手牌
                    hand_cards = self.detect_hand_cards(screen)
                    self.current_state['hand_cards'] = hand_cards
                    
                    # 决策动作
                    action = self.decide_action(target_card, hand_cards, player_states)
                elif is_one_more:
                    action = self.decide_action()
                        
                # 执行动作
                if is_my_turn or is_one_more:
                    self.execute_action(action)
                    time.sleep(2)  # 等待动作完成
                    times += 1
                
                # 等待下一次检测
                time.sleep(self.config['detection_interval'])
                
            except KeyboardInterrupt:
                print("检测循环已停止")
                break
            except Exception as e:
                print(f"检测循环中出现错误: {e}")
                time.sleep(1)
    
    def create_template_directories(self):
        """创建模板目录"""
        os.makedirs(self.config['card_templates_dir'], exist_ok=True)
        os.makedirs(self.config['player_templates_dir'], exist_ok=True)
        print(f"已创建模板目录: {self.config['card_templates_dir']}, {self.config['player_templates_dir']}")
    
    def save_current_state(self, filename: str = 'game_state.json'):
        """保存当前游戏状态"""
        state_data = {
            'target_card': self.current_state['target_card'].value if self.current_state['target_card'] else None,
            'hand_cards_count': len(self.current_state['hand_cards']),
            'player_states': {k: v.value for k, v in self.current_state['player_states'].items()},
            'is_my_turn': self.current_state['is_my_turn']
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        print(f"游戏状态已保存到: {filename}")

def main():
    """主程序入口"""
    print("=== 自动化打牌系统 ===")
    
    # 创建游戏自动化实例
    game_auto = GameAutomation()
    
    # 创建模板目录
    game_auto.create_template_directories()
    
    print("\n请确保：")
    print("1. 已将卡牌模板图像放置在 'card_templates' 目录下")
    print("2. 已将玩家状态模板图像放置在 'player_templates' 目录下")
    print("3. 游戏窗口已打开处于全屏状态，并已进入一局新的游戏")
    print("4. 按 Ctrl+C 可停止程序")
    
    input("\n准备就绪后按回车开始...")
    print("5秒后开始...")

    time.sleep(5)
    
    # 开始检测循环
    game_auto.run_detection_loop()

if __name__ == "__main__":
    main()