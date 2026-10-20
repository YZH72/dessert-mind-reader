import numpy as np

# ========================================================================
# 7 種情緒在 3D 情感感官空間中的標準座標 (學術基準點)
# 
# 軸向定義 (根據論文手稿)：
# X 軸：愉悅度 (Valence)    [-1.0 (極度痛苦/高熱量需求) 到 +1.0 (極度快樂/精緻輕盈)]
# Y 軸：喚醒度 (Arousal)    [-1.0 (昏沉低落/入口即化) 到 +1.0 (情緒激昂/硬脆咀嚼)]
# Z 軸：敏感度 (Sensitivity)[-1.0 (焦慮不安/溫和包覆) 到 +1.0 (厭惡排斥/濃烈刺激)]
# ========================================================================
EMOTION_3D_SPACE = {
    "happy":    np.array([ 0.8,  0.6, -0.3]), 
    "neutral":  np.array([ 0.0,  0.0,  0.0]), 
    "surprise": np.array([ 0.4,  0.7, -0.1]), 
    "sad":      np.array([-0.6, -0.5, -0.6]), 
    "fear":     np.array([-0.9, -0.4, -0.8]), 
    "angry":    np.array([-0.5,  0.8,  0.2]), 
    "disgust":  np.array([-0.4,  0.3,  0.8])  
}

class EmotionSpaceTracker:
    def __init__(self):
        self.counts = {k: 0 for k in EMOTION_3D_SPACE.keys()}
        # ========================================================================
        # 3D 甜點資料庫 (目前共 25 款經典甜點)
        # 座標定義: [X: 豐富度(熱量/罪惡感), Y: 軟硬度(咀嚼感), Z: 風味防線(溫和/刺激)]
        # 範圍均為 -1.0 到 +1.0
        # 擴充成分 (ingredients) 與標籤 (tags) 以支援過敏原及偏好計算
        # ========================================================================
        self.dessert_db = {
            # === 【區塊一：左下立體區】高熱量補償、入口即化/極度綿密、溫和包覆 (強烈對應 Sad / Fear) ===
            "濃郁熔岩巧克力蛋糕": {
                "coord": np.array([-0.9, -0.8, -0.7]),
                "ingredients": {"cocoa", "egg", "milk", "gluten"},
                "tags": {"chocolate", "warm", "rich"}
            },
            "經典巧克力慕斯杯": {
                "coord": np.array([-0.7, -0.7, -0.6]),
                "ingredients": {"cocoa", "milk", "egg"},
                "tags": {"chocolate", "soft", "rich"}
            },
            "日式焦糖烤布丁": {
                "coord": np.array([-0.5, -0.8, -0.5]),
                "ingredients": {"egg", "milk"},
                "tags": {"caramel", "soft"}
            },
            "香草卡士達千層蛋糕": {
                "coord": np.array([-0.5, -0.6, -0.4]),
                "ingredients": {"milk", "egg", "gluten"},
                "tags": {"vanilla", "layered", "soft"}
            },
            "義式微醺提拉米蘇": {
                "coord": np.array([-0.6, -0.7, -0.3]),
                "ingredients": {"milk", "egg", "alcohol", "gluten"},
                "tags": {"coffee", "creamy", "boozy", "italian"}
            }, # 帶一點點可可與酒香的慰藉
            "美式重乳酪蛋糕": {
                "coord": np.array([-0.8, -0.5, -0.5]),
                "ingredients": {"milk", "egg", "butter", "cream_cheese", "graham_cracker"},
                "tags": {"cheese", "rich", "dense"}
            }, # 質地綿密、高乳酪脂質補償
            "流心巴斯克乳酪蛋糕": {
                "coord": np.array([-0.7, -0.6, -0.4]),
                "ingredients": {"milk", "egg", "butter", "cream_cheese"},
                "tags": {"cheese", "molten_center", "dense"}
            }, # 焦香、內層流心
            "英式熱鮮奶茶司康": {
                "coord": np.array([-0.6, -0.4, -0.5]),
                "ingredients": {"milk", "gluten", "butter"},
                "tags": {"tea", "warm", "classic"}
            }, 

            # === 【區塊二：左上立體區】高熱量補償、極度硬脆、風味厚實/高刺激 (強烈對應 Angry / Disgust) ===
            "厚切堅果燕麥脆餅": {
                "coord": np.array([-0.4, 0.9, 0.3]),
                "ingredients": {"nuts", "gluten", "oats"},
                "tags": {"crunchy", "nuts"}
            },
            "比利時脆粒黑巧克力": {
                "coord": np.array([-0.6, 0.7, 0.5]),
                "ingredients": {"cocoa", "milk"},
                "tags": {"chocolate", "crunchy"}
            },
            "焦糖核桃肉桂塔": {
                "coord": np.array([-0.5, 0.8, 0.4]),
                "ingredients": {"nuts", "cinnamon", "gluten"},
                "tags": {"cinnamon", "nuts", "crunchy"}
            },
            "義式硬脆餅": {
                "coord": np.array([-0.3, 0.9, 0.4]),
                "ingredients": {"nuts", "gluten"},
                "tags": {"crunchy", "nuts", "classic"}
            },
            "法式可麗露": {
                "coord": np.array([-0.4, 0.6, 0.3]),
                "ingredients": {"egg", "milk", "gluten", "alcohol", "caramel_shell"},
                "tags": {"caramel", "chewy", "crispy"}
            }, # 外皮硬脆焦香、內裡帶嚼勁
            "極致 85% 苦甜巧克力": {
                "coord": np.array([-0.3, 0.3, 0.9]),
                "ingredients": {"cocoa"},
                "tags": {"chocolate", "bittersweet"}
            }, # 強烈苦味用以抽離厭惡
            "黑糖核桃熱肉桂捲": {
                "coord": np.array([-0.7, 0.5, 0.5]),
                "ingredients": {"nuts", "cinnamon", "gluten"},
                "tags": {"cinnamon", "warm", "nuts"}
            }, # 雖然軟，但辛辣肉桂與高甜適合憤怒宣洩

            # === 【區塊三：右上立體區】精緻輕盈、層次脆感、風味明亮 (強烈對應 Happy / Surprise) ===
            "法式繽紛馬卡龍": {
                "coord": np.array([0.8, 0.6, -0.2]),
                "ingredients": {"almond", "egg"},
                "tags": {"sweet", "fruity", "colorful", "crispy"}
            },
            "草莓鮮奶油戚風蛋糕": {
                "coord": np.array([0.7, -0.3, -0.3]),
                "ingredients": {"strawberry", "milk", "egg", "gluten"},
                "tags": {"fruity", "soft", "airy", "strawberry"}
            },
            "跳跳糖香草冰淇淋": {
                "coord": np.array([0.5, 0.7, 0.5]),
                "ingredients": {"milk", "sugar"},
                "tags": {"poprock", "vanilla", "cold"}
            },
            "炙燒檸檬蛋白霜塔": {
                "coord": np.array([0.6, 0.4, 0.6]),
                "ingredients": {"lemon", "egg", "gluten"},
                "tags": {"sour", "crispy", "citrus"}
            },
            "法式精品鹽可頌": {
                "coord": np.array([0.3, 0.6, 0.7]),
                "ingredients": {"butter", "gluten"},
                "tags": {"salty", "crispy", "chewy"}
            }, # [X:偏輕盈享受, Y:外焦脆內嚼勁, Z:灑鹽風味衝突驚喜]
            "青蘋果樂園派": {
                "coord": np.array([0.5, 0.5, 0.5]),
                "ingredients": {"apple", "gluten", "sugar", "butter"},
                "tags": {"sour", "fruity", "crispy"}
            }, # 酸甜爽脆
            "酥皮芒果泡芙": {
                "coord": np.array([0.6, 0.3, 0.2]),
                "ingredients": {"mango", "milk", "egg", "gluten"},
                "tags": {"fruity", "soft", "airy"}
            }, # 酥脆外皮與多汁水果

            # === 【區塊四：中心平穩區】常溫、中等質地、平衡風味 (強烈對應 Neutral) ===
            "原味法式瑪德蓮": {
                "coord": np.array([0.0, 0.0, 0.0]),
                "ingredients": {"butter", "egg", "gluten"},
                "tags": {"classic"}
            },
            "經典老奶奶檸檬蛋糕": {
                "coord": np.array([0.1, 0.1, 0.2]),
                "ingredients": {"lemon", "gluten", "egg"},
                "tags": {"citrus", "sour", "classic"}
            },
            "英式伯爵茶磅蛋糕": {
                "coord": np.array([0.0, 0.2, -0.1]),
                "ingredients": {"tea", "gluten", "butter"},
                "tags": {"tea", "classic"}
            } #淡淡茶香，質地扎實但溫和
        }

    def record_frame_emotion(self, emotion_label):
        if emotion_label in self.counts:
            self.counts[emotion_label] += 1

    
    def calculate_user_coordinate(self):
        total_frames = sum(self.counts.values())# 算出看鏡頭的總時間（總幀數）
        if total_frames == 0:
            return np.array([0.0, 0.0, 0.0])
        
        #計算綜合情緒座標，根據各情緒的幀數比例加權平均出一個綜合座標   
        user_vector = np.zeros(3)
        for emo, count in self.counts.items():
            weight = count / total_frames  # 情緒權重（比例）
            user_vector += EMOTION_3D_SPACE[emo] * weight #  權重去抓取/乘上標準座標，最後加總成一個綜合座標
            
        return user_vector

    def get_top_k_recommendations(self, user_vector=None, user_preferences=None, user_allergens=None, axis_weights=(1.0, 1.0, 1.0), k=3):
        """
        多模態推薦演算法：
        1. 硬性過敏原剔除
        2. 軟性偏好優先級分池（有偏好標籤優先，內部依 3D 情緒距離排序）
        3. 無偏好時直接依 3D 情緒距離排序
        """
        # 如果呼叫時沒有傳入 user_vector，則自動依據現有 counts 計算
        if user_vector is None:
            user_vector = self.calculate_user_coordinate()
        # 確保型態轉為 set 方便比對與運算
        user_preferences = set(user_preferences) if user_preferences else set()
        user_allergens = set(user_allergens) if user_allergens else set()
        w = np.array(axis_weights)

        pref_desserts = []   # 符合偏好的甜點 (名稱, 距離分數)
        other_desserts = []  # 其他一般甜點 (名稱, 距離分數)

        for dessert_name, data in self.dessert_db.items():
            ingredients = data.get("ingredients", set())
            tags = data.get("tags", set())

            # 1. 硬性過敏原剔除
            if user_allergens.intersection(ingredients):
                continue

            # 2. 計算 3D 歐氏距離基礎分 (0 ~ 1 之間，越近越高分)
            dessert_vector = data["coord"]
            weighted_sq_dist = np.sum(w * ((user_vector - dessert_vector) ** 2))
            weighted_dist = np.sqrt(weighted_sq_dist)
            base_score = 1.0 / (1.0 + weighted_dist)

            # 3. 判斷是否符合偏好 (關鍵字/標籤模糊比對)
            is_preferred = False
            if user_preferences:
                for pref in user_preferences:
                    pref_str = str(pref).lower()
                    tag_match = any(pref_str in str(tag).lower() or str(tag).lower() in pref_str for tag in tags)
                    name_match = pref_str in dessert_name.lower()
                    if tag_match or name_match:
                        is_preferred = True
                        break

            # 4. 分池存放
            if is_preferred:
                pref_desserts.append((dessert_name, base_score))
            else:
                other_desserts.append((dessert_name, base_score))

        # 兩池分別依 3D 情緒距離分數由高到低排序
        pref_desserts.sort(key=lambda x: x[1], reverse=True)
        other_desserts.sort(key=lambda x: x[1], reverse=True)

        # 合併結果：偏好池排前面，一般池補在後面
        final_ranked = pref_desserts + other_desserts

        # 取前 K 個甜點名稱
        top_k_desserts = [dessert[0] for dessert in final_ranked[:k]]
        return top_k_desserts

    def get_current_weights(self):
        total = sum(self.counts.values())
        if total == 0: 
            return {k: 0.0 for k in self.counts.keys()}
        return {k: round(v / total, 2) for k, v in self.counts.items()}

    def reset_space(self):
        self.counts = {k: 0 for k in EMOTION_3D_SPACE.keys()}

# ========================================================================
# 對外決策接口 (一次推薦 3 款甜點)
# ========================================================================
def get_recommendation_by_emotions(emotion_counts, user_allergens=None, user_preferences=None, axis_weights=(1.0, 1.0, 1.0)):
    """
    接收前端傳來的各情緒幀數統計，回傳推薦的 3 款甜點名稱列表
    emotion_counts: 格式例如 {"happy": 10, "neutral": 5, ...}
    """
    tracker = EmotionSpaceTracker()
    # 將傳入的計數塞進 tracker
    tracker.counts = emotion_counts
    
    # 修正：必須先根據計數算出綜合情緒座標 user_vector！
    user_vector = tracker.calculate_user_coordinate()
    # k 為 3，撈出 3D 空間中距離最近的前 3 名甜點
    recommendations = tracker.get_top_k_recommendations(
        user_vector=user_vector,
        k=3,
        user_allergens=user_allergens,
        user_preferences=user_preferences,
        axis_weights=axis_weights
    )
    
    # 回傳整個陣列 (例如: ["法式繽紛馬卡龍", "草莓鮮奶油戚風蛋糕", "酥皮芒果泡芙"])
    return recommendations