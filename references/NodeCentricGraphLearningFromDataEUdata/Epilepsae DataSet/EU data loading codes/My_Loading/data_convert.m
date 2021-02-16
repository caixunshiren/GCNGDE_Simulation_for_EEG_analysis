function [y, conv_sizes] = data_convert(x, load_core)
conversion_names = load_core.data_conversions;
[y, conv_sizes] = conversion_names(1).my_apply(x, load_core);
end

% function [y, conv_sizes] = data_convert(x, load_core)
% conversion_names = load_core.data_conversions;
% conv_sizes = [];
% y = cell(size(x,1), size(x,2));
% for i=1:length(conversion_names)
%     [conv, new_conv_sizes] = conversion_names(i).my_apply(x, load_core);
%     y((i-1)*size(x,1)+1:i*size(x,1)) = num2cell(conv, [2,3,4]);
%     conv_sizes = [conv_sizes  new_conv_sizes];
% end
% end
