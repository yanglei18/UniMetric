import numpy as np

class VLMScaleEstimator:
    def __init__(self, vision_model):
        self.vision_model = vision_model
    
    def generate_dimension_range(self, image_path, object_type):
        """
        Call VLM model to get real-world dimension range for a specific object type.
        
        Args:
            image_path (str): Path to the image
            object_type (str): Type of the object (e.g., "Car", "Pedestrian", "Cyclist")
            
        Returns:
            dict: Real-world dimension range with keys like "length", "width", "height"
        """
        # Design prompt template based on object type
        prompt_template = {
            "Car": "What is the typical real-world dimension range for a standard car in meters? "
                   "Please provide the range for length, width, and height separately. "
                   "Format the output as: length: [min, max], width: [min, max], height: [min, max]",
            "Bus": "What is the typical real-world dimension range for a bus in meters? "
                   "Please provide the range for length, width, and height separately. "
                   "Format the output as: length: [min, max], width: [min, max], height: [min, max]",
            "Pedestrian": "What is the typical real-world dimension range for a human pedestrian in meters? "
                          "Please provide the range for height (and optionally width) separately. "
                          "Format the output as: height: [min, max], width: [min, max]",
            "Cyclist": "What is the typical real-world dimension range for a cyclist (including bicycle) in meters? "
                       "Please provide the range for length, width, and height separately. "
                       "Format the output as: length: [min, max], width: [min, max], height: [min, max]",
            "default": "What is the typical real-world dimension range for a {object_type} in meters? "
                       "Please provide the range for length, width, and height separately. "
                       "Format the output as: length: [min, max], width: [min, max], height: [min, max]"
        }
        
        # Get appropriate prompt based on object type
        prompt = prompt_template.get(object_type, prompt_template["default"]).format(object_type=object_type)
        
        # Call VLM model to get dimension range
        try:
            response = self.vision_model.generate(
                text=prompt,
                data_path=image_path,
                system_text="You are a physical world expert. Please provide accurate dimension ranges."
            )
            return self.parse_vlm_response(response, object_type)
        except Exception as e:
            print(f"Error getting dimension range from VLM: {e}")
            return None
    
    def parse_vlm_response(self, response, object_type):
        """
        Parse VLM response to extract dimension ranges.
        
        Args:
            response (str): Raw response from VLM model
            object_type (str): Type of the object
            
        Returns:
            dict: Parsed dimension ranges with keys like "length", "width", "height"
        """
        import re
        
        parsed = {}
        
        # Define regex patterns to extract ranges
        patterns = {
            "length": r"length\s*:\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
            "width": r"width\s*:\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]",
            "height": r"height\s*:\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]"
        }
        
        # Extract each dimension range
        for dim, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                min_val = float(match.group(1))
                max_val = float(match.group(2))
                parsed[dim] = [min_val, max_val]
        
        # Handle specific object types with default values if needed
        if object_type == "Pedestrian":
            if "height" not in parsed:
                parsed["height"] = [1.5, 1.8]
            if "width" not in parsed:
                parsed["width"] = [0.4, 0.6]
            # Pedestrians don't have a meaningful length, so set to same as width
            parsed["length"] = parsed["width"]
        
        return parsed
    
    def calculate_scale_intervals(self, object_list, dimension_ranges):
        """
        Calculate scale factor intervals for each object based on their relative dimensions and VLM-derived real dimensions.
        
        Args:
            object_list (list): List of objects with relative dimensions
            dimension_ranges (dict): Real-world dimension ranges from VLM
            
        Returns:
            list: List of scale intervals for each object
        """
        scale_intervals = []
        
        for obj in object_list:
            obj_type = obj["type"]
            relative_dim = obj["relative_dimension"]  # [length, width, height]
            
            if obj_type not in dimension_ranges:
                continue
            
            real_ranges = dimension_ranges[obj_type]
            
            # Calculate scale intervals for each dimension
            intervals = []
            for i, dim_name in enumerate(["length", "width", "height"]):
                if dim_name in real_ranges and relative_dim[i] > 0:
                    real_min, real_max = real_ranges[dim_name]
                    rel_dim = relative_dim[i]
                    # Scale interval: s ∈ [real_min/rel_dim, real_max/rel_dim]
                    intervals.append([real_min / rel_dim, real_max / rel_dim])
            
            if intervals:
                # Combine intervals: take intersection of all dimension intervals
                combined_min = max(interval[0] for interval in intervals)
                combined_max = min(interval[1] for interval in intervals)
                
                if combined_min <= combined_max:
                    scale_intervals.append([combined_min, combined_max])
        
        return scale_intervals
    
    def search_consistent_scale(self, scale_intervals, num_candidates=100):
        """
        Search for the scale factor that is consistent with the most objects.
        
        Args:
            scale_intervals (list): List of scale intervals for each object
            num_candidates (int): Number of candidate scale factors to evaluate
            
        Returns:
            tuple: (best_scale, best_count) - the scale factor that covers the most objects
        """
        if not scale_intervals:
            return 1.0, 0
        
        # Determine the global range to search
        global_min = min(interval[0] for interval in scale_intervals)
        global_max = max(interval[1] for interval in scale_intervals)
        
        if global_min >= global_max:
            return global_min, len(scale_intervals)
        
        # Generate candidate scale factors
        candidates = np.linspace(global_min, global_max, num_candidates)
        
        # Count how many intervals each candidate is in
        best_count = 0
        best_scale = global_min
        
        for s in candidates:
            count = sum(1 for interval in scale_intervals if interval[0] <= s <= interval[1])
            if count > best_count:
                best_count = count
                best_scale = s
        
        return best_scale, best_count
    
    def optimize_scale(self, scale_intervals, initial_scale, num_steps=100):
        """
        Optimize the scale factor within the consistent interval using gradient descent.
        
        Args:
            scale_intervals (list): List of scale intervals for each object
            initial_scale (float): Initial scale factor from consistency search
            num_steps (int): Number of optimization steps
            
        Returns:
            float: Optimized scale factor
        """
        if not scale_intervals:
            return initial_scale
        
        def interval_loss(s):
            """Calculate loss for a given scale factor s."""
            loss = 0.0
            for interval in scale_intervals:
                a, b = interval
                if s < a:
                    loss += (s - a) ** 2
                elif s > b:
                    loss += (s - b) ** 2
                # Else, loss is 0
            return loss
        
        # Use golden section search for optimization
        def golden_section_search(f, a, b, tol=1e-5):
            gr = (np.sqrt(5) + 1) / 2
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            
            while abs(b - a) > tol:
                if f(c) < f(d):
                    b = d
                else:
                    a = c
                c = b - (b - a) / gr
                d = a + (b - a) / gr
            
            return (a + b) / 2
        
        # Determine the optimization range
        global_min = min(interval[0] for interval in scale_intervals)
        global_max = max(interval[1] for interval in scale_intervals)
        
        # Perform optimization
        optimized_scale = golden_section_search(interval_loss, global_min, global_max)
        
        return optimized_scale
    
    def estimate_metric_scale(self, object_list, image_path, scale_pairs=None):
        """
        Main function to estimate metric scale using VLM and optimization.
        
        Args:
            object_list (list): List of objects with type, relative position, and relative dimension
            image_path (str): Path to the image file
            scale_pairs (list, optional): List of (relative_length, absolute_length) pairs for joint estimation
            
        Returns:
            float: Estimated metric scale factor
        """
        # Get unique object types
        object_types = list(set(obj["type"] for obj in object_list))
        
        # Get dimension ranges from VLM for each object type
        dimension_ranges = {}
        for obj_type in object_types:
            dim_range = self.generate_dimension_range(image_path, obj_type)
            if dim_range:
                dimension_ranges[obj_type] = dim_range
        
        # Calculate scale intervals for each object from VLM
        vlm_scale_intervals = self.calculate_scale_intervals(object_list, dimension_ranges)
        
        # Add scale intervals from provided scale pairs
        scale_intervals = vlm_scale_intervals.copy()
        
        if scale_pairs:
            print(f"Using {len(scale_pairs)} scale pairs for joint estimation")
            # Convert scale pairs to intervals (assuming small uncertainty around absolute length)
            for rel_len, abs_len in scale_pairs:
                if rel_len > 0:
                    # Add small uncertainty (1% around absolute length)
                    uncertainty = 0.01 * abs_len
                    min_abs = abs_len - uncertainty
                    max_abs = abs_len + uncertainty
                    # Calculate scale interval: s ∈ [min_abs/rel_len, max_abs/rel_len]
                    scale_interval = [min_abs / rel_len, max_abs / rel_len]
                    scale_intervals.append(scale_interval)
        
        if not scale_intervals:
            print("No valid scale intervals found, returning default scale 1.0")
            return 1.0
        
        # Search for consistent scale
        consistent_scale, consistent_count = self.search_consistent_scale(scale_intervals)
        # Calculate how many VLM intervals the consistent scale covers
        vlm_consistent_count = sum(1 for interval in vlm_scale_intervals if interval[0] <= consistent_scale <= interval[1])
        
        # Optimize scale within the consistent interval
        optimized_scale = self.optimize_scale(scale_intervals, consistent_scale)
        
        final_scale = optimized_scale
        print(f"Scale estimation results:")
        print(f"- Consistent scale: {consistent_scale:.4f} (covers {consistent_count}/{len(scale_intervals)} total, {vlm_consistent_count}/{len(vlm_scale_intervals)} VLM intervals)")        
        return final_scale