classdef customExpDecayLearnRate < deep.LearnRateSchedule
    % customExpDecayLearnRate Custom exponential decay learning rate schedule
    % based on the function f(x) = 100e^{-(g(x-1))^2}
    % where g(x) = l(e^{(-x+100)/m} - k)

    properties
        % Schedule properties
        l
        k
        m
    end

    methods
        function schedule = customExpDecayLearnRate()
            % customExpDecayLearnRate Custom exponential decay learning rate
            % schedule with pre-calculated parameters
            
            % Set schedule properties
            schedule.FrequencyUnit = "epoch";
            schedule.NumSteps = Inf;
            
            % Pre-calculated parameters from your equations
            x1 = (1 + sqrt(1 + 4*sqrt(3*log(10)))) / 2;
            schedule.l = 1;
            schedule.k = x1;
            schedule.m = 50 / log(schedule.k);
        end

        function [schedule,learnRate] = update(schedule,initialLearnRate,epoch,~)
            % UPDATE Update learning rate schedule
            %   [schedule,learnRate] = update(schedule,initialLearnRate,epoch,~)
            %   calculates the learning rate for the specified epoch
            %   using the custom exponential decay function

            % Extract parameters
            l = schedule.l;
            k = schedule.k;
            m = schedule.m;
            
            % Calculate g(x-1) where x = epoch
            g_val = l * (exp((-(epoch-1) + 100)/m) - k);
            
            % Calculate learning rate using f(x) = 100e^{-(g(x-1))^2}
            learnRate = initialLearnRate * 100 * exp(-(g_val)^2);
        end
    end
end